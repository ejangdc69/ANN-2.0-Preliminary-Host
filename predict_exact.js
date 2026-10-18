/** Exact 35→12→1 inference from saved JSON. No weather inputs are used. */
export function predictFromModel(model, {month, activity, location}) {
  const inputs = {month, activity: String(activity), location};
  const z = model.one_hot_features.map(({input, category}) => {
    if (!model.categories[input].includes(String(inputs[input]))) {
      throw new Error(`Unknown ${input}: ${inputs[input]}`);
    }
    return Number(String(inputs[input]) === String(category));
  });
  const preActivations = model.b_hidden.map((bias,j) =>
    bias + z.reduce((sum,zi,i) => sum + zi * model.W_input_hidden[i][j],0));
  const activations = preActivations.map(x => Math.max(0,x));
  const contributions = activations.map((h,j) => h * model.W_hidden_output[j]);
  const days = contributions.reduce((a,b)=>a+b,model.b_output);
  return {days, preActivations, activations, contributions, outputBias:model.b_output};
}

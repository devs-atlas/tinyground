import nextMDX from "@next/mdx";

const withMDX = nextMDX();

const nextConfig = {
  pageExtensions: ["js", "jsx", "mdx", "ts", "tsx"],
  experimental: {
    forceSwcTransforms: true,
  },
};

export default withMDX(nextConfig);

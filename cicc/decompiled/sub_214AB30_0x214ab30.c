// Function: sub_214AB30
// Address: 0x214ab30
//
__int64 sub_214AB30()
{
  void *v0; // rax
  void *v1; // rax

  v0 = sub_214A9D0();
  sub_16D4020((__int64)v0, (__int64)"nvptx", (__int64)"NVIDIA PTX 32-bit", (__int64)"NVPTX", (__int64)sub_214A9B0, 0);
  v1 = sub_214AA80();
  return sub_16D4020(
           (__int64)v1,
           (__int64)"nvptx64",
           (__int64)"NVIDIA PTX 64-bit",
           (__int64)"NVPTX",
           (__int64)sub_214A9C0,
           0);
}

// Function: sub_301FFA0
// Address: 0x301ffa0
//
__int64 sub_301FFA0()
{
  void *v0; // rax
  void *v1; // rax

  v0 = sub_301FDE0();
  sub_C0D4B0((__int64)v0, (__int64)"nvptx", (__int64)"NVIDIA PTX 32-bit", (__int64)"NVPTX", (__int64)sub_301FDC0, 0);
  v1 = sub_301FEC0();
  return sub_C0D4B0(
           (__int64)v1,
           (__int64)"nvptx64",
           (__int64)"NVIDIA PTX 64-bit",
           (__int64)"NVPTX",
           (__int64)sub_301FDD0,
           0);
}

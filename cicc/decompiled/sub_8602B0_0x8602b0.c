// Function: sub_8602B0
// Address: 0x8602b0
//
_BYTE *__fastcall sub_8602B0(__int64 a1)
{
  __int64 v1; // rdi

  sub_860260(a1);
  v1 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *(_BYTE *)(v1 + 8) |= 8u;
  return sub_732EF0(v1);
}

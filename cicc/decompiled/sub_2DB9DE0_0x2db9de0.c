// Function: sub_2DB9DE0
// Address: 0x2db9de0
//
__int64 __fastcall sub_2DB9DE0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 *v4; // rdi

  v2 = a1 + 3;
  v4 = a1 + 1;
  *v4 = (__int64)v2;
  *(v4 - 1) = a2;
  v4[1] = 0x800000000LL;
  *((_DWORD *)v4 + 12) = 0;
  sub_3157150(v4, 0);
  a1[8] = (__int64)(a1 + 10);
  a1[9] = 0x400000000LL;
  return sub_2DB9B50(a1);
}

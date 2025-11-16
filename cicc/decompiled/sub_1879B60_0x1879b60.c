// Function: sub_1879B60
// Address: 0x1879b60
//
__int64 __fastcall sub_1879B60(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (__int64)(0xCCCCCCCCCCCCCCCDLL * ((__int64)&a2[-a1] >> 4) + 1) / 2;
  v7 = 80 * v6;
  v10 = a1 + 80 * v6;
  if ( v6 <= a4 )
  {
    sub_1877DB0(a1, (char *)(a1 + 80 * v6), a3);
    sub_1877DB0(v10, a2, a3);
  }
  else
  {
    sub_1879B60(a1, a1 + 80 * v6, a3);
    sub_1879B60(v10, a2, a3);
  }
  sub_18790B0(
    a1,
    v10,
    (__int64)a2,
    0xCCCCCCCCCCCCCCCDLL * (v7 >> 4),
    0xCCCCCCCCCCCCCCCDLL * ((__int64)&a2[-v10] >> 4),
    a3,
    a4);
  return v9;
}

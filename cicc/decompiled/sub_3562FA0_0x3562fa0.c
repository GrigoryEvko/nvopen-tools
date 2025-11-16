// Function: sub_3562FA0
// Address: 0x3562fa0
//
__int64 __fastcall sub_3562FA0(char *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]

  v5 = (0x2E8BA2E8BA2E8BA3LL * ((a2 - (__int64)a1) >> 3) + 1) / 2;
  v6 = 88 * v5;
  v7 = (__int64)&a1[88 * v5];
  if ( v5 <= a4 )
  {
    sub_3540EB0((__int64)a1, (__int64)&a1[88 * v5], a3);
    sub_3540EB0(v7, a2, a3);
  }
  else
  {
    sub_3562FA0(a1, &a1[88 * v5], a3);
    sub_3562FA0(v7, a2, a3);
  }
  sub_3562590(a1, v7, a2, 0x2E8BA2E8BA2E8BA3LL * (v6 >> 3), 0x2E8BA2E8BA2E8BA3LL * ((a2 - v7) >> 3), a3, a4);
  return v9;
}

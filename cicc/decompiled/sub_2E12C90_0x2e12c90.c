// Function: sub_2E12C90
// Address: 0x2e12c90
//
__int64 __fastcall sub_2E12C90(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 *i; // [rsp+8h] [rbp-38h]

  sub_2E1DCC0(a1[6], *a1, a1[4], a1[5], a1 + 7);
  result = (__int64)&a3[a4];
  for ( i = (__int64 *)result; i != a3; result = sub_2E20270(a1[6], a2, v10, 0, a5, a6) )
    v10 = *a3++;
  return result;
}

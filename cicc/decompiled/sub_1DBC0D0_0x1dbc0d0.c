// Function: sub_1DBC0D0
// Address: 0x1dbc0d0
//
__int64 __fastcall sub_1DBC0D0(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 *i; // [rsp+8h] [rbp-38h]

  sub_1DC3BD0(a1[36], a1[29], a1[34], a1[35], a1 + 37);
  result = (__int64)&a3[a4];
  for ( i = (__int64 *)result; i != a3; result = sub_1DC5C40(a1[36], a2, v10, 0, a5, a6) )
    v10 = *a3++;
  return result;
}

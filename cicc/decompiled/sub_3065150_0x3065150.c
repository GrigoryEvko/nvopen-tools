// Function: sub_3065150
// Address: 0x3065150
//
unsigned __int64 __fastcall sub_3065150(__int64 a1, __int64 *a2, int a3, unsigned int a4, __int64 *a5)
{
  __int64 v8; // r13
  unsigned __int64 v9; // rbx
  signed __int64 v10; // rcx
  unsigned __int64 result; // rax
  __int64 v12; // [rsp+0h] [rbp-50h]
  unsigned __int64 v13; // [rsp+8h] [rbp-48h]
  unsigned __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-38h]

  v12 = sub_BCDA70(a2, a4);
  v8 = sub_BCDA70(a2, a4 * a3);
  sub_C4DEC0((__int64)&v14, (__int64)a5, a4, 0);
  v9 = sub_3064F80(a1 + 8, v12, (__int64 *)&v14, 0, 1);
  v10 = sub_3064F80(a1 + 8, v8, a5, 1, 0);
  result = v10 + v9;
  if ( __OFADD__(v10, v9) )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v10 <= 0 )
      result = 0x8000000000000000LL;
  }
  if ( v15 > 0x40 )
  {
    if ( v14 )
    {
      v13 = result;
      j_j___libc_free_0_0(v14);
      return v13;
    }
  }
  return result;
}

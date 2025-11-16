// Function: sub_B6E160
// Address: 0xb6e160
//
__int64 __fastcall sub_B6E160(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r14
  char *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // [rsp+8h] [rbp-58h]
  __m128i v14; // [rsp+10h] [rbp-50h] BYREF
  __int64 v15; // [rsp+20h] [rbp-40h] BYREF

  v6 = sub_B6DC00(*a1, a2, a3);
  v7 = v6;
  if ( a4 )
  {
    sub_B6E0E0(&v14, a2, a3, a4, a1, v6);
    sub_BA8CA0(a1, v14.m128i_i64[0], v14.m128i_i64[1], v7);
    result = v12;
    if ( (__int64 *)v14.m128i_i64[0] != &v15 )
    {
      v13 = v12;
      j_j___libc_free_0(v14.m128i_i64[0], v15 + 1);
      return v13;
    }
  }
  else
  {
    v8 = sub_B60C10(a2);
    sub_BA8CA0(a1, v8, v9, v7);
    return v10;
  }
  return result;
}

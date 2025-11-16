// Function: sub_2E8F150
// Address: 0x2e8f150
//
__int64 __fastcall sub_2E8F150(__int64 a1, __int64 a2)
{
  unsigned __int16 *v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rax
  unsigned __int16 *v8; // rbx
  unsigned __int16 *v9; // r14
  __int32 v10; // eax
  __int64 result; // rax
  unsigned __int16 *v12; // rbx
  unsigned __int16 *i; // r14
  __int32 v14; // eax
  __m128i v15; // [rsp+0h] [rbp-60h] BYREF
  __int64 v16; // [rsp+10h] [rbp-50h]
  __int64 v17; // [rsp+18h] [rbp-48h]
  __int64 v18; // [rsp+20h] [rbp-40h]

  v4 = *(unsigned __int16 **)(a1 + 16);
  v5 = *((unsigned int *)v4 + 3);
  v6 = *((unsigned __int8 *)v4 + 8);
  v7 = 4 * (5LL * *v4 + 5);
  v8 = &v4[v7 + v5 + v6];
  v9 = &v8[*((unsigned __int8 *)v4 + 9)];
  if ( v9 != v8 )
  {
    do
    {
      v10 = *v8++;
      v15.m128i_i64[0] = 805306368;
      v16 = 0;
      v15.m128i_i32[2] = v10;
      v17 = 0;
      v18 = 0;
      sub_2E8EAD0(a1, a2, &v15);
    }
    while ( v9 != v8 );
    v4 = *(unsigned __int16 **)(a1 + 16);
    v5 = *((unsigned int *)v4 + 3);
    v6 = *((unsigned __int8 *)v4 + 8);
    v7 = 4 * (5LL * *v4 + 5);
  }
  result = v5 + v7;
  v12 = &v4[result];
  for ( i = &v12[v6]; i != v12; result = sub_2E8EAD0(a1, a2, &v15) )
  {
    v14 = *v12++;
    v15.m128i_i64[0] = 0x20000000;
    v16 = 0;
    v15.m128i_i32[2] = v14;
    v17 = 0;
    v18 = 0;
  }
  return result;
}

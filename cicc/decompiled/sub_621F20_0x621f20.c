// Function: sub_621F20
// Address: 0x621f20
//
__int64 __fastcall sub_621F20(__m128i *a1, const __m128i *a2, int a3, _BOOL4 *a4)
{
  __m128i *v7; // r12
  int v8; // r8d
  __m128i *v9; // rbx
  __int128 *v10; // r10
  __int64 v11; // r9
  _WORD *v12; // rcx
  __int64 v13; // rdi
  __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rax
  __m128i v17; // kr00_16
  int v18; // ebx
  __int64 result; // rax
  int v20; // [rsp+Ch] [rbp-84h]
  _BOOL4 v21; // [rsp+1Ch] [rbp-74h] BYREF
  __m128i v22; // [rsp+20h] [rbp-70h] BYREF
  __m128i v23; // [rsp+30h] [rbp-60h] BYREF
  __int128 v24; // [rsp+40h] [rbp-50h] BYREF
  __m128i v25; // [rsp+50h] [rbp-40h]

  v7 = (__m128i *)a2;
  v24 = 0;
  v25 = 0;
  if ( a3 )
  {
    v8 = 0;
    v9 = a1;
    if ( a1->m128i_i16[0] >= 0 )
    {
      if ( a2->m128i_i16[0] >= 0 )
        goto LABEL_6;
      goto LABEL_4;
    }
    v9 = &v22;
    v22 = _mm_loadu_si128(a1);
    sub_621710(v22.m128i_i16, a4);
    v8 = 1;
    if ( a2->m128i_i16[0] < 0 )
    {
LABEL_4:
      v7 = &v23;
      v20 = v8;
      v23 = _mm_loadu_si128(a2);
      sub_621710(v23.m128i_i16, a4);
      v8 = v20 ^ 1;
    }
  }
  else
  {
    v8 = 0;
    v9 = a1;
  }
LABEL_6:
  v10 = (__int128 *)((char *)&v24 + 14);
  v11 = 7;
  v12 = (_WORD *)&v24 + 7;
  do
  {
    v13 = v7->m128i_u16[v11];
    v14 = 0;
    v15 = 0;
    do
    {
      v16 = v15 + (unsigned __int16)v12[v14 + 8] + v13 * v9->m128i_u16[v14 + 7];
      v12[v14 + 8] = v16;
      --v14;
      v15 = v16 >> 16;
    }
    while ( v14 != -8 );
    --v11;
    *v12-- = WORD1(v16);
  }
  while ( v11 != -1 );
  v17 = v25;
  v18 = 0;
  *a1 = v25;
  while ( 1 )
  {
    result = (__int64)v10 - 2;
    if ( *(_WORD *)v10 )
      v18 = 1;
    if ( &v24 == v10 )
      break;
    v10 = (__int128 *)((char *)v10 - 2);
  }
  if ( a3 )
  {
    if ( v8 )
      result = sub_621710(a1->m128i_i16, &v21);
    else
      v21 = 0;
    if ( v17.m128i_i16[0] < 0 )
    {
      result = 1;
      if ( !v21 )
        v18 = 1;
    }
  }
  *a4 = v18;
  return result;
}

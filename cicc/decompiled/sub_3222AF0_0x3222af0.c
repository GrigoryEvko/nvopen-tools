// Function: sub_3222AF0
// Address: 0x3222af0
//
__m128i *__fastcall sub_3222AF0(__m128i *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v6; // rdx
  unsigned __int8 *v7; // r13
  unsigned __int64 v8; // r14
  __m128i v9; // xmm1
  _QWORD *v10; // rdi
  int v11; // eax
  __int64 v12; // rdx
  int v13; // eax
  int v14; // esi
  __int8 *v15; // rsi
  __int8 *v16; // rax
  __m128i *v17; // rdx
  __int8 v18; // cl
  __m128i v19; // xmm0
  __m128i v20; // [rsp+0h] [rbp-50h] BYREF
  _BYTE *v21; // [rsp+10h] [rbp-40h] BYREF
  __int64 v22; // [rsp+18h] [rbp-38h]
  _BYTE v23[48]; // [rsp+20h] [rbp-30h] BYREF

  if ( (unsigned __int16)sub_3220AA0(a2) <= 4u
    || !*(_BYTE *)(a3 + 32)
    || (v7 = (unsigned __int8 *)sub_B91420(*(_QWORD *)(a3 + 24)), v8 = v6, *(_DWORD *)(a3 + 16) != 1) )
  {
    a1[1].m128i_i8[0] = 0;
    return a1;
  }
  v22 = 0;
  v21 = v23;
  v23[0] = 0;
  if ( !v6 )
  {
    v9 = _mm_loadu_si128(&v20);
    a1[1].m128i_i8[0] = 1;
    *a1 = v9;
    return a1;
  }
  sub_22410F0((unsigned __int64 *)&v21, (v6 + 1) >> 1, 0);
  v10 = v21;
  if ( (v8 & 1) == 0 )
    goto LABEL_11;
  v11 = (__int16)word_3F64060[*v7];
  if ( v11 != -1 )
  {
    *v21 = v11;
    --v8;
    ++v7;
    v10 = (_QWORD *)((char *)v10 + 1);
LABEL_11:
    if ( v8 >> 1 )
    {
      v12 = 0;
      do
      {
        v13 = (__int16)word_3F64060[v7[2 * v12]];
        v14 = (__int16)word_3F64060[v7[2 * v12 + 1]];
        if ( v13 == -1 )
          break;
        if ( v14 == -1 )
          break;
        *((_BYTE *)v10 + v12++) = v14 | (16 * v13);
      }
      while ( v8 >> 1 != v12 );
    }
    v10 = v21;
  }
  if ( v22 > 0 )
  {
    v15 = (char *)v10 + v22;
    v16 = (__int8 *)v10;
    v17 = &v20;
    do
    {
      v18 = *v16++;
      v17 = (__m128i *)((char *)v17 + 1);
      v17[-1].m128i_i8[15] = v18;
    }
    while ( v16 != v15 );
  }
  v19 = _mm_loadu_si128(&v20);
  a1[1].m128i_i8[0] = 1;
  *a1 = v19;
  if ( v10 != (_QWORD *)v23 )
    j_j___libc_free_0((unsigned __int64)v10);
  return a1;
}

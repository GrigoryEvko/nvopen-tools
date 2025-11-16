// Function: sub_1D9BC60
// Address: 0x1d9bc60
//
__int64 __fastcall sub_1D9BC60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  const __m128i *v7; // rsi
  const __m128i *v8; // r14
  int v9; // r10d
  const __m128i *v10; // rax
  const __m128i *v11; // r9
  int v12; // edx
  __int64 v13; // r8
  __int64 v14; // rdi
  __int64 v15; // r13
  _WORD *v16; // r15
  unsigned __int16 v17; // cx
  __int16 *v18; // r8
  _WORD *v19; // r15
  int v20; // edx
  unsigned __int16 *v21; // r13
  unsigned int v22; // r15d
  unsigned int i; // edi
  bool v24; // cf
  __int16 *v25; // r15
  __int16 v26; // r8
  char v27; // al
  int v28; // edi
  __int64 v31; // [rsp+8h] [rbp-B0h]
  __int64 v32; // [rsp+10h] [rbp-A8h]
  char v33; // [rsp+1Fh] [rbp-99h]
  __int64 v34; // [rsp+20h] [rbp-98h]
  __m128i v35; // [rsp+28h] [rbp-90h]
  __m128i v36; // [rsp+58h] [rbp-60h]

  v34 = a4;
  v32 = a4 + 8 * a5;
  if ( a4 == v32 )
    goto LABEL_31;
  v33 = 0;
  while ( 1 )
  {
    v7 = *(const __m128i **)(*(_QWORD *)v34 + 32LL);
    v8 = (const __m128i *)((char *)v7 + 40 * *(unsigned int *)(*(_QWORD *)v34 + 40LL));
    if ( v7 != v8 )
      break;
LABEL_22:
    v27 = v33;
LABEL_27:
    v34 += 8;
    if ( v34 == v32 )
    {
      if ( v27 )
      {
        *(_BYTE *)a1 = 1;
        *(_BYTE *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v31;
        return a1;
      }
LABEL_31:
      *(_BYTE *)a1 = 1;
      *(_BYTE *)(a1 + 16) = 0;
      return a1;
    }
  }
LABEL_6:
  while ( 1 )
  {
    v9 = v7->m128i_i32[2];
    v35 = _mm_loadu_si128(v7);
    if ( !v7->m128i_i8[0] )
    {
      if ( v9 )
      {
        v10 = *(const __m128i **)(a3 + 32);
        v11 = (const __m128i *)((char *)v10 + 40 * *(unsigned int *)(a3 + 40));
        if ( v10 != v11 )
          break;
      }
    }
    v7 = (const __m128i *)((char *)v7 + 40);
    if ( v8 == v7 )
      goto LABEL_22;
  }
  while ( 1 )
  {
    v12 = v10->m128i_i32[2];
    v36 = _mm_loadu_si128(v10);
    if ( v10->m128i_i8[0] || !v12 )
      goto LABEL_20;
    if ( v9 != v12 )
    {
      if ( v9 < 0 || v12 < 0 )
        goto LABEL_20;
      v13 = *(_QWORD *)(a2 + 240);
      v14 = *(_QWORD *)(v13 + 8);
      v15 = *(_QWORD *)(v13 + 56);
      LODWORD(v13) = *(_DWORD *)(v14 + 24LL * (unsigned int)v9 + 16);
      v16 = (_WORD *)(v15 + 2LL * ((unsigned int)v13 >> 4));
      v17 = *v16 + v9 * (v13 & 0xF);
      v18 = v16 + 1;
      LODWORD(v16) = *(_DWORD *)(v14 + 24LL * (unsigned int)v12 + 16);
      v20 = ((unsigned __int8)v16 & 0xF) * v12;
      v19 = (_WORD *)(v15 + 2LL * ((unsigned int)v16 >> 4));
      LOWORD(v20) = *v19 + v20;
      v21 = v19 + 1;
      v22 = v17;
      for ( i = (unsigned __int16)v20; ; i = (unsigned __int16)v20 )
      {
        v24 = v22 < i;
        if ( v22 == i )
          break;
        while ( v24 )
        {
          v25 = v18 + 1;
          v26 = *v18;
          v17 += v26;
          if ( !v26 )
            goto LABEL_20;
          v18 = v25;
          v22 = v17;
          v24 = v17 < i;
          if ( v17 == i )
            goto LABEL_19;
        }
        v28 = *v21;
        if ( !(_WORD)v28 )
          goto LABEL_20;
        v20 += v28;
        ++v21;
      }
    }
LABEL_19:
    if ( ((v36.m128i_i8[3] | v35.m128i_i8[3]) & 0x10) != 0 )
      break;
LABEL_20:
    v10 = (const __m128i *)((char *)v10 + 40);
    if ( v11 == v10 )
    {
      v7 = (const __m128i *)((char *)v7 + 40);
      if ( v8 != v7 )
        goto LABEL_6;
      goto LABEL_22;
    }
  }
  if ( !v33 )
  {
    v33 = 1;
    v31 = v34;
    v27 = 1;
    goto LABEL_27;
  }
  *(_BYTE *)a1 = 0;
  *(_BYTE *)(a1 + 16) = 0;
  return a1;
}

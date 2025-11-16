// Function: sub_7A4160
// Address: 0x7a4160
//
__int64 __fastcall sub_7A4160(__int64 a1, __int64 a2, __int64 *a3, FILE *a4)
{
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r12
  int v9; // ebx
  __int64 v10; // r13
  unsigned __int64 v11; // r14
  __int64 v12; // rsi
  char i; // cl
  unsigned int v14; // edx
  __int64 v15; // rax
  __int64 v16; // r10
  __int64 v17; // rdi
  _WORD *v18; // rdi
  __int64 v19; // rax
  unsigned int v20; // eax
  __int64 *v21; // r14
  const __m128i *v22; // rbx
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 *v25; // rsi
  __int64 v26; // r14
  const __m128i *v27; // r13
  const __m128i *v28; // rdx
  __m128i *v29; // rax
  __int64 v30; // rbx
  __int64 v32; // rdx
  __int64 *v34; // [rsp+8h] [rbp-68h]
  __int64 v35; // [rsp+8h] [rbp-68h]
  __int64 v36; // [rsp+10h] [rbp-60h]
  int v38; // [rsp+20h] [rbp-50h]
  const __m128i *v39; // [rsp+20h] [rbp-50h]
  __int64 v40; // [rsp+20h] [rbp-50h]
  unsigned int v41; // [rsp+28h] [rbp-48h]
  int v42[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v5 = sub_76FF70(a2);
  if ( !v5 )
    goto LABEL_34;
  v36 = 0;
  v8 = v7;
  v9 = 0;
  v10 = v6;
  v41 = 0;
  v38 = 0;
  v34 = a3;
  v11 = v5;
  do
  {
    v12 = *(_QWORD *)(v11 + 120);
    for ( i = *(_BYTE *)(v12 + 140); i == 12; i = *(_BYTE *)(v12 + 140) )
      v12 = *(_QWORD *)(v12 + 160);
    v14 = qword_4F08388 & (v11 >> 3);
    v15 = qword_4F08380 + 16LL * v14;
    v16 = *(_QWORD *)v15;
    if ( v11 == *(_QWORD *)v15 )
    {
LABEL_37:
      v17 = *(unsigned int *)(v15 + 8);
      if ( i == 6 )
        goto LABEL_38;
    }
    else
    {
      while ( v16 )
      {
        v14 = qword_4F08388 & (v14 + 1);
        v15 = qword_4F08380 + 16LL * v14;
        v16 = *(_QWORD *)v15;
        if ( *(_QWORD *)v15 == v11 )
          goto LABEL_37;
      }
      v17 = 0;
      if ( i == 6 )
      {
LABEL_38:
        if ( v38 )
          goto LABEL_34;
        v32 = *(_QWORD *)(v10 + v17 + 24);
        v36 = v10 + v17;
        if ( (*(_BYTE *)(v32 - 9) & 1) == 0
          && ((unsigned __int8)(1 << ((*(_QWORD *)(v10 + v17) - v32) & 7))
            & *(_BYTE *)(v32 + -((((unsigned int)*(_QWORD *)(v10 + v17) - (unsigned int)v32) >> 3) + 10))) == 0 )
        {
          goto LABEL_42;
        }
        v38 = 1;
        goto LABEL_17;
      }
    }
    if ( i != 2 || v9 > 1 )
      goto LABEL_34;
    v18 = (_WORD *)(v10 + v17);
    if ( ((unsigned __int8)(1 << (((_BYTE)v18 - v8) & 7))
        & *(_BYTE *)(v8 + -(((unsigned int)((_DWORD)v18 - v8) >> 3) + 10))) == 0 )
      goto LABEL_42;
    v19 = *(unsigned __int8 *)(v12 + 160);
    v42[0] = 0;
    v20 = sub_620EE0(v18, byte_4B6DF90[v19], v42);
    if ( !v9 || v41 > v20 )
      v41 = v20;
    ++v9;
LABEL_17:
    v11 = sub_76FF70(*(_QWORD *)(v11 + 112));
  }
  while ( v11 );
  v21 = v34;
  if ( !v38 || v9 != 2 )
  {
LABEL_34:
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xD27u, (FILE *)(a1 + 112), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
    }
    return 0;
  }
  if ( (*(_BYTE *)(v36 + 8) & 8) == 0 )
  {
LABEL_42:
    sub_6855B0(0xD2Au, a4, (_QWORD *)(a1 + 96));
    return 0;
  }
  v22 = *(const __m128i **)v36;
  v23 = *(_QWORD *)(v36 + 24);
  v24 = v34[1];
  if ( v34[2] > 0 )
    v34[2] = 0;
  if ( v41 > v24 )
  {
    v40 = *v34;
    v35 = sub_823970(24LL * v41);
    sub_823A00(v40, 24 * v24);
    v21[1] = v41;
    *v21 = v35;
  }
  if ( (int)v41 > 0 )
  {
    v25 = v21;
    v26 = v23;
    v27 = v22;
    v28 = (const __m128i *)((char *)v22 + 24 * v41);
    while ( (*(_BYTE *)(v36 + 8) & 2) == 0
         && ((unsigned __int8)(1 << (((_BYTE)v27 - v26) & 7))
           & *(_BYTE *)(v26 + -(((unsigned int)((_DWORD)v27 - v26) >> 3) + 10))) != 0 )
    {
      v30 = v25[2];
      if ( v30 == v25[1] )
      {
        v39 = v28;
        sub_7A3E20((const __m128i **)v25);
        v28 = v39;
      }
      v29 = (__m128i *)(*v25 + 24 * v30);
      if ( v29 )
      {
        *v29 = _mm_loadu_si128(v27);
        v29[1].m128i_i64[0] = v27[1].m128i_i64[0];
      }
      v27 = (const __m128i *)((char *)v27 + 24);
      v25[2] = v30 + 1;
      if ( v28 == v27 )
        return 1;
    }
    goto LABEL_42;
  }
  return 1;
}

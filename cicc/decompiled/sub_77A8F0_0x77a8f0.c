// Function: sub_77A8F0
// Address: 0x77a8f0
//
_DWORD *__fastcall sub_77A8F0(__int64 a1, char *a2, __int64 a3, _DWORD *a4)
{
  __int64 v4; // r12
  __m128i *v6; // rax
  int v7; // edi
  __m128i *v8; // r14
  unsigned __int64 v9; // rcx
  unsigned int i; // edx
  _QWORD *v11; // rax
  unsigned __int64 v12; // r8
  _DWORD *result; // rax
  __int64 v14; // r13
  unsigned int v15; // r15d
  unsigned int v16; // edx
  unsigned int v17; // eax
  unsigned int v18; // esi
  size_t v19; // rdx
  char *v20; // r8
  size_t v21; // rax
  unsigned __int64 v22; // r15
  char j; // al
  __int64 v24; // rax
  unsigned __int8 *v25; // r15
  unsigned int v26; // eax
  _WORD *v27; // r14
  int v28; // r12d
  _QWORD *v29; // rbx
  unsigned __int64 v30; // rax
  unsigned int v31; // ecx
  unsigned int v32; // ecx
  __int64 v33; // rsi
  unsigned int v34; // edx
  __m128i *v35; // rax
  __m128i v36; // xmm0
  __m128i *v37; // rax
  int v38; // eax
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r8
  __int64 v42; // rax
  _QWORD *v43; // rax
  __int64 v44; // rax
  __int64 v45; // [rsp+0h] [rbp-80h]
  __int64 v46; // [rsp+8h] [rbp-78h]
  __m128i *v47; // [rsp+10h] [rbp-70h]
  __int64 v48; // [rsp+18h] [rbp-68h]
  unsigned int v49; // [rsp+18h] [rbp-68h]
  unsigned int v50; // [rsp+18h] [rbp-68h]
  unsigned int v51; // [rsp+24h] [rbp-5Ch]
  __int64 v52; // [rsp+28h] [rbp-58h]
  __int64 v53; // [rsp+28h] [rbp-58h]
  _QWORD *v54; // [rsp+28h] [rbp-58h]
  _QWORD *v55; // [rsp+28h] [rbp-58h]
  __int64 v56; // [rsp+30h] [rbp-50h]
  _DWORD v58[13]; // [rsp+4Ch] [rbp-34h] BYREF

  v4 = a1;
  v58[0] = 1;
  v6 = sub_740F80(a2);
  v7 = *(_DWORD *)(a1 + 8);
  v8 = v6;
  v9 = v6[11].m128i_u64[1];
  v56 = v6[11].m128i_i64[0];
  for ( i = v7 & (v9 >> 3); ; i = v7 & (i + 1) )
  {
    v11 = (_QWORD *)(*(_QWORD *)v4 + 16LL * i);
    if ( v9 == *v11 )
      break;
    if ( !*v11 )
      goto LABEL_9;
  }
  v12 = v11[1];
  if ( !v12 )
  {
LABEL_9:
    v14 = v8[8].m128i_i64[0];
    v15 = 16;
    if ( (unsigned __int8)(*(_BYTE *)(v14 + 140) - 2) > 1u )
      v15 = sub_7764B0(v4, v8[8].m128i_u64[0], v58);
    if ( !v58[0] )
      goto LABEL_7;
    if ( (unsigned __int8)(*(_BYTE *)(v14 + 140) - 8) > 3u )
    {
      v52 = 16;
      v17 = 16;
    }
    else
    {
      v16 = (v15 + 7) >> 3;
      v17 = v16 + 9;
      if ( (((_BYTE)v16 + 9) & 7) != 0 )
        v17 = v16 + 17 - (((_BYTE)v16 + 9) & 7);
      v52 = v17;
    }
    if ( (v15 & 7) != 0 )
      v15 = v15 + 8 - (v15 & 7);
    v18 = v15 + v17;
    v19 = v15 + v17 + 16;
    if ( (*(_BYTE *)(v4 + 132) & 8) == 0 )
    {
      v43 = (_QWORD *)qword_4F082A0;
      if ( qword_4F082A0 )
      {
        qword_4F082A0 = *(_QWORD *)(qword_4F082A0 + 8);
      }
      else
      {
        v43 = (_QWORD *)sub_823970(0x10000);
        v19 = v18 + 16;
      }
      *v43 = *(_QWORD *)(v4 + 152);
      *(_QWORD *)(v4 + 152) = v43;
      v43[1] = 0;
      v44 = *(_QWORD *)(v4 + 152);
      *(_BYTE *)(v4 + 132) |= 8u;
      *(_QWORD *)(v4 + 160) = 0;
      *(_QWORD *)(v4 + 144) = v44 + 24;
      *(_QWORD *)(v4 + 176) = 0;
      *(_DWORD *)(v4 + 168) = 0;
    }
    if ( (unsigned int)v19 > 0x400 )
    {
      v49 = v19;
      v40 = sub_822B10(v18 + 32);
      v19 = v49;
      v41 = v40;
      v42 = *(_QWORD *)(v4 + 160);
      *(_DWORD *)(v41 + 8) = v18 + 32;
      *(_QWORD *)v41 = v42;
      *(_DWORD *)(v41 + 12) = *(_DWORD *)(v4 + 168);
      *(_QWORD *)(v4 + 160) = v41;
      v20 = (char *)(v41 + 16);
    }
    else
    {
      v20 = *(char **)(v4 + 144);
      v21 = v18 + 24 - (v19 & 7);
      if ( (v19 & 7) == 0 )
        v21 = v19;
      if ( 0x10000 - (*(_DWORD *)(v4 + 144) - *(_DWORD *)(v4 + 152)) < (unsigned int)v21 )
      {
        v50 = v19;
        v51 = v21;
        sub_772E70((_QWORD *)(v4 + 144));
        v20 = *(char **)(v4 + 144);
        v19 = v50;
        v21 = v51;
      }
      *(_QWORD *)(v4 + 144) = &v20[v21];
    }
    v12 = (unsigned __int64)memset(v20, 0, v19) + v52;
    *(_DWORD *)(v12 + v15) = 0;
    *(_QWORD *)(v12 - 8) = v14;
    if ( (unsigned __int8)(*(_BYTE *)(v14 + 140) - 9) <= 2u )
      *(_QWORD *)v12 = 0;
    if ( !v58[0] )
      goto LABEL_7;
    v22 = *(_QWORD *)(v14 + 160);
    for ( j = *(_BYTE *)(v22 + 140); j == 12; j = *(_BYTE *)(v22 + 140) )
      v22 = *(_QWORD *)(v22 + 160);
    if ( (unsigned __int8)(j - 2) > 1u )
    {
      v55 = (_QWORD *)v12;
      v26 = sub_7764B0(v4, v22, v58);
      v12 = (unsigned __int64)v55;
      if ( !v58[0] )
        goto LABEL_7;
      v39 = *(_QWORD *)(v22 + 128);
      v25 = (unsigned __int8 *)v8[11].m128i_i64[1];
      v53 = v39;
    }
    else
    {
      v24 = *(_QWORD *)(v22 + 128);
      v25 = (unsigned __int8 *)v8[11].m128i_i64[1];
      v53 = v24;
      v26 = 16;
    }
    if ( (_DWORD)v56 )
    {
      v47 = v8;
      v48 = v26;
      v27 = (_WORD *)v12;
      v46 = v4;
      v28 = 0;
      v45 = a3;
      v29 = (_QWORD *)v12;
      do
      {
        ++v28;
        v30 = sub_722AB0(v25, v53);
        sub_620D80(v27, v30);
        v25 += v53;
        v31 = (_DWORD)v27 - (_DWORD)v29;
        v27 = (_WORD *)((char *)v27 + v48);
        *((_BYTE *)v29 - (v31 >> 3) - 10) |= 1 << (v31 & 7);
      }
      while ( (_DWORD)v56 != v28 );
      v12 = (unsigned __int64)v29;
      v8 = v47;
      v4 = v46;
      a3 = v45;
    }
    *(_BYTE *)(v12 - 9) |= 1u;
    v32 = *(_DWORD *)(v4 + 8);
    v33 = *(_QWORD *)v4;
    v34 = v32 & (v12 >> 3);
    v35 = (__m128i *)(*(_QWORD *)v4 + 16LL * v34);
    if ( v35->m128i_i64[0] )
    {
      v36 = _mm_loadu_si128(v35);
      v35->m128i_i64[0] = v12;
      v35->m128i_i64[1] = (__int64)v8;
      do
      {
        v34 = v32 & (v34 + 1);
        v37 = (__m128i *)(v33 + 16LL * v34);
      }
      while ( v37->m128i_i64[0] );
      *v37 = v36;
    }
    else
    {
      v35->m128i_i64[0] = v12;
      v35->m128i_i64[1] = (__int64)v8;
    }
    v38 = *(_DWORD *)(v4 + 12) + 1;
    *(_DWORD *)(v4 + 12) = v38;
    if ( 2 * v38 > v32 )
    {
      v54 = (_QWORD *)v12;
      sub_7704A0(v4);
      v12 = (unsigned __int64)v54;
    }
  }
  if ( !v58[0] )
  {
LABEL_7:
    *a4 = 0;
    return a4;
  }
  *(_QWORD *)a3 = v12;
  *(_OWORD *)(a3 + 8) = 0;
  *(_BYTE *)(a3 + 8) = 72;
  result = (_DWORD *)*(unsigned __int8 *)(a3 + 8);
  *(_QWORD *)(a3 + 24) = v12;
  *(_DWORD *)(a3 + 8) = (unsigned int)result | ((_DWORD)v56 << 8);
  return result;
}

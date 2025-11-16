// Function: sub_7A0A10
// Address: 0x7a0a10
//
__int64 __fastcall sub_7A0A10(
        unsigned int a1,
        const __m128i *a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6)
{
  const __m128i *v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // r8
  int v15; // edi
  int v16; // eax
  _QWORD *v17; // r15
  __int64 v18; // rsi
  __int64 v19; // r11
  char v20; // al
  unsigned int v21; // eax
  unsigned int v22; // r15d
  int v24; // eax
  _QWORD **v25; // rsi
  const __m128i *v26; // r12
  __int64 *v27; // rbx
  __int32 v28; // edi
  __int32 v29; // esi
  __int64 v30; // rcx
  __int64 v31; // r12
  unsigned int v32; // edx
  _DWORD *m; // rax
  __int64 v34; // rcx
  __int64 v35; // rsi
  unsigned int v36; // eax
  __int64 v37; // rdx
  __int64 v38; // r13
  unsigned __int64 v39; // r8
  unsigned __int64 v40; // rsi
  char i; // al
  size_t v42; // r11
  __int32 v43; // esi
  unsigned int j; // edx
  __int64 v45; // rax
  void *v46; // rdi
  __int64 v47; // rsi
  __int64 v48; // rdx
  _QWORD *v49; // rcx
  _QWORD *k; // rax
  __int64 v51; // rax
  unsigned int v52; // eax
  unsigned __int64 v53; // [rsp+0h] [rbp-80h]
  __int64 v54; // [rsp+8h] [rbp-78h]
  __int64 v55; // [rsp+8h] [rbp-78h]
  unsigned __int64 v56; // [rsp+8h] [rbp-78h]
  __int64 v58; // [rsp+18h] [rbp-68h]
  __int32 v59; // [rsp+24h] [rbp-5Ch]
  __int64 v60; // [rsp+28h] [rbp-58h]
  __int64 v61; // [rsp+30h] [rbp-50h]
  __int64 v62; // [rsp+38h] [rbp-48h]
  int v63[13]; // [rsp+4Ch] [rbp-34h] BYREF

  v9 = a2;
  v10 = a2[4].m128i_u32[0];
  v61 = a2[1].m128i_i64[0];
  v62 = a2[1].m128i_i64[1];
  v60 = a2[2].m128i_i64[0];
  v59 = a2[2].m128i_i32[2];
  v58 = a2[3].m128i_i64[0];
  v11 = (unsigned int)(a2[8].m128i_i32[0] + 1);
  a2[8].m128i_i32[0] = v11;
  a2[2].m128i_i32[2] = v11;
  v12 = v10 & v11;
  v13 = a2[3].m128i_i64[1];
  v14 = v13 + 4LL * ((unsigned int)v10 & (unsigned int)v11);
  v15 = *(_DWORD *)v14;
  *(_DWORD *)v14 = v11;
  if ( v15 )
  {
    do
    {
      v12 = v10 & (v12 + 1);
      v11 = v13 + 4LL * v12;
      v14 = *(unsigned int *)v11;
    }
    while ( (_DWORD)v14 );
    *(_DWORD *)v11 = v15;
    v24 = v9[4].m128i_i32[1] + 1;
    v9[4].m128i_i32[1] = v24;
    if ( 2 * v24 <= (unsigned int)v10 )
    {
LABEL_3:
      v9[3].m128i_i64[0] = 0;
      if ( a1 )
        goto LABEL_4;
LABEL_14:
      v25 = (_QWORD **)a3;
      v22 = 1;
      goto LABEL_24;
    }
  }
  else
  {
    v16 = v9[4].m128i_i32[1] + 1;
    v9[4].m128i_i32[1] = v16;
    if ( 2 * v16 <= (unsigned int)v10 )
      goto LABEL_3;
  }
  sub_7702C0((__int64)&v9[3].m128i_i64[1]);
  v9[3].m128i_i64[0] = 0;
  if ( !a1 )
    goto LABEL_14;
LABEL_4:
  v17 = *(_QWORD **)(a3 + 56);
  v18 = v17[3];
  v19 = v17[1];
  if ( v18 )
  {
    v20 = *(_BYTE *)(v18 + 40);
    if ( v20 != 20 )
    {
      v54 = v17[1];
      if ( v20 )
        sub_721090();
      v21 = sub_795660((__int64)v9, v18, v11, v10, v14, a6);
      v19 = v54;
      a1 = v21;
      if ( !v54 )
      {
        if ( !v21 )
          return 0;
        goto LABEL_23;
      }
      if ( !v21 )
        return 0;
      goto LABEL_22;
    }
    if ( *(_QWORD *)(v18 + 72) )
    {
      v53 = a5;
      v26 = v9;
      v27 = *(__int64 **)(v18 + 72);
      v55 = v17[1];
      do
      {
        if ( *((_BYTE *)v27 + 8) == 7 && !(unsigned int)sub_7A0470(v26, v27[2], 0, (FILE *)(v27[2] + 64)) )
          break;
        v27 = (__int64 *)*v27;
      }
      while ( v27 );
      v9 = v26;
      v19 = v55;
      a5 = v53;
    }
  }
  if ( v19 )
  {
LABEL_22:
    v56 = *(_QWORD *)(v19 + 8);
    if ( !(unsigned int)sub_7A0470(v9, v56, 0, (FILE *)(v56 + 64)) )
    {
      v39 = v56;
      v40 = *(_QWORD *)(v56 + 120);
      for ( i = *(_BYTE *)(v40 + 140); i == 12; i = *(_BYTE *)(v40 + 140) )
        v40 = *(_QWORD *)(v40 + 160);
      v63[0] = 1;
      v42 = 16;
      if ( (unsigned __int8)(i - 2) > 1u )
      {
        v52 = sub_7764B0((__int64)v9, v40, v63);
        v39 = v56;
        v42 = v52;
      }
      v43 = v9->m128i_i32[2];
      for ( j = v43 & (v39 >> 3); ; j = v43 & (j + 1) )
      {
        v45 = v9->m128i_i64[0] + 16LL * j;
        v46 = *(void **)v45;
        if ( v39 == *(_QWORD *)v45 )
          break;
        if ( !v46 )
          goto LABEL_55;
      }
      v46 = *(void **)(v45 + 8);
LABEL_55:
      memset(v46, 0, v42);
    }
  }
LABEL_23:
  v25 = (_QWORD **)v17[2];
  v22 = a1;
LABEL_24:
  if ( !(unsigned int)sub_786210((__int64)v9, v25, a5, (char *)a5) )
    return 0;
  if ( ((*(_BYTE *)(a3 + 25) & 3) != 0 || *(_BYTE *)(a4 + 140) == 6) && (*(_BYTE *)(a5 + 8) & 4) != 0 )
  {
    v47 = *(_QWORD *)(a5 + 16);
    v48 = 2;
    v49 = *(_QWORD **)v47;
    for ( k = **(_QWORD ***)v47; k; ++v48 )
    {
      v49 = k;
      k = (_QWORD *)*k;
    }
    *v49 = qword_4F08088;
    *(_BYTE *)(a5 + 8) &= ~4u;
    v51 = *(_QWORD *)(v47 + 24);
    qword_4F08080 += v48;
    qword_4F08088 = v47;
    *(_QWORD *)(a5 + 16) = v51;
  }
  if ( v9[3].m128i_i64[0] )
    v22 = sub_799890((__int64)v9);
  v28 = v9[2].m128i_i32[2];
  v29 = v9[4].m128i_i32[0];
  v30 = v9[3].m128i_i64[1];
  v31 = v9[2].m128i_i64[0];
  v32 = v29 & v28;
  for ( m = (_DWORD *)(v30 + 4LL * (v29 & (unsigned int)v28)); v28 != *m; m = (_DWORD *)(v30 + 4LL * v32) )
    v32 = v29 & (v32 + 1);
  *m = 0;
  if ( *(_DWORD *)(v30 + 4LL * ((v32 + 1) & v29)) )
    sub_771390(v9[3].m128i_i64[1], v9[4].m128i_i32[0], v32);
  --v9[4].m128i_i32[1];
  v9[1].m128i_i64[0] = v61;
  v9[2].m128i_i32[2] = v59;
  v9[1].m128i_i64[1] = v62;
  v9[3].m128i_i64[0] = v58;
  v9[2].m128i_i64[0] = v60;
  if ( v31 && v60 != v31 )
  {
    while ( 1 )
    {
      v34 = *(unsigned int *)(v31 + 12);
      v35 = v9[3].m128i_i64[1];
      v36 = v34 & v9[4].m128i_i32[0];
      v37 = *(unsigned int *)(v35 + 4LL * v36);
      if ( (_DWORD)v37 == (_DWORD)v34 || !(_DWORD)v34 )
        break;
      while ( (_DWORD)v37 )
      {
        v36 = v9[4].m128i_i32[0] & (v36 + 1);
        v37 = *(unsigned int *)(v35 + 4LL * v36);
        if ( (_DWORD)v34 == (_DWORD)v37 )
          goto LABEL_37;
      }
      v38 = *(_QWORD *)v31;
      sub_822B90(v31, *(unsigned int *)(v31 + 8), v37, v34);
      if ( !v38 )
      {
        v31 = 0;
        break;
      }
      v31 = v38;
    }
LABEL_37:
    v9[2].m128i_i64[0] = v31;
  }
  return v22;
}

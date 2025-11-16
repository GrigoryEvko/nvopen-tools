// Function: sub_2FA9100
// Address: 0x2fa9100
//
void __fastcall sub_2FA9100(__int64 a1, __int64 a2, __int64 *i, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  bool v8; // zf
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int64 v17; // rsi
  const __m128i *v18; // rdi
  unsigned __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // rcx
  __m128i *v22; // rdx
  const __m128i *v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  const __m128i *v26; // rcx
  unsigned __int64 v27; // r13
  __int64 v28; // rax
  unsigned __int64 v29; // rdi
  __m128i *v30; // rdx
  const __m128i *v31; // rax
  __int64 v32; // rcx
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rsi
  _QWORD *v36; // rax
  __int64 v37; // r15
  __int64 j; // rax
  __int64 v39; // rdx
  __int64 v40; // r14
  _QWORD *v41; // rax
  char v42; // si
  char v43; // dl
  __int64 v44; // [rsp+8h] [rbp-248h] BYREF
  __m128i v45; // [rsp+10h] [rbp-240h] BYREF
  char v46; // [rsp+20h] [rbp-230h]
  __int64 v47; // [rsp+30h] [rbp-220h] BYREF
  _QWORD *v48; // [rsp+38h] [rbp-218h]
  unsigned int v49; // [rsp+40h] [rbp-210h]
  unsigned int v50; // [rsp+44h] [rbp-20Ch]
  char v51; // [rsp+4Ch] [rbp-204h]
  char v52[64]; // [rsp+50h] [rbp-200h] BYREF
  __int64 v53; // [rsp+90h] [rbp-1C0h] BYREF
  __int64 v54; // [rsp+98h] [rbp-1B8h]
  unsigned __int64 v55; // [rsp+A0h] [rbp-1B0h]
  char v56[8]; // [rsp+B0h] [rbp-1A0h] BYREF
  unsigned __int64 v57; // [rsp+B8h] [rbp-198h]
  char v58; // [rsp+CCh] [rbp-184h]
  char v59[64]; // [rsp+D0h] [rbp-180h] BYREF
  unsigned __int64 v60; // [rsp+110h] [rbp-140h]
  unsigned __int64 v61; // [rsp+118h] [rbp-138h]
  unsigned __int64 v62; // [rsp+120h] [rbp-130h]
  _QWORD v63[3]; // [rsp+130h] [rbp-120h] BYREF
  char v64; // [rsp+14Ch] [rbp-104h]
  const __m128i *v65; // [rsp+190h] [rbp-C0h]
  unsigned __int64 v66; // [rsp+198h] [rbp-B8h]
  char v67[8]; // [rsp+1A8h] [rbp-A8h] BYREF
  unsigned __int64 v68; // [rsp+1B0h] [rbp-A0h]
  char v69; // [rsp+1C4h] [rbp-8Ch]
  const __m128i *v70; // [rsp+208h] [rbp-48h]
  const __m128i *v71; // [rsp+210h] [rbp-40h]

  v6 = a1;
  v8 = *(_BYTE *)(a2 + 28) == 0;
  v44 = a1;
  if ( v8 )
    goto LABEL_8;
  v9 = *(__int64 **)(a2 + 8);
  a4 = *(unsigned int *)(a2 + 20);
  for ( i = &v9[a4]; i != v9; ++v9 )
  {
    if ( a1 == *v9 )
      return;
  }
  if ( (unsigned int)a4 >= *(_DWORD *)(a2 + 16) )
  {
LABEL_8:
    sub_C8CC70(a2, a1, (__int64)i, a4, a1, a6);
    if ( !(_BYTE)i )
      return;
  }
  else
  {
    v10 = (unsigned int)(a4 + 1);
    *(_DWORD *)(a2 + 20) = v10;
    *i = a1;
    ++*(_QWORD *)a2;
  }
  sub_24FFF20(v63, &v44, (__int64)i, v10, v6, a6);
  sub_C8CD80((__int64)&v47, (__int64)v52, (__int64)v63, v11, v12, v13);
  v17 = v66;
  v18 = v65;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v19 = v66 - (_QWORD)v65;
  if ( (const __m128i *)v66 == v65 )
  {
    v21 = 0;
  }
  else
  {
    if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_88;
    v20 = sub_22077B0(v66 - (_QWORD)v65);
    v17 = v66;
    v18 = v65;
    v21 = v20;
  }
  v53 = v21;
  v54 = v21;
  v55 = v21 + v19;
  if ( v18 != (const __m128i *)v17 )
  {
    v22 = (__m128i *)v21;
    v23 = v18;
    do
    {
      if ( v22 )
      {
        *v22 = _mm_loadu_si128(v23);
        v15 = v23[1].m128i_i64[0];
        v22[1].m128i_i64[0] = v15;
      }
      v23 = (const __m128i *)((char *)v23 + 24);
      v22 = (__m128i *)((char *)v22 + 24);
    }
    while ( v23 != (const __m128i *)v17 );
    v21 += 8 * ((unsigned __int64)((char *)&v23[-2].m128i_u64[1] - (char *)v18) >> 3) + 24;
  }
  v54 = v21;
  v18 = (const __m128i *)v56;
  sub_C8CD80((__int64)v56, (__int64)v59, (__int64)v67, v21, v15, v16);
  v26 = v71;
  v17 = (unsigned __int64)v70;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v27 = (char *)v71 - (char *)v70;
  if ( v71 != v70 )
  {
    if ( v27 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v28 = sub_22077B0((char *)v71 - (char *)v70);
      v26 = v71;
      v17 = (unsigned __int64)v70;
      v29 = v28;
      goto LABEL_21;
    }
LABEL_88:
    sub_4261EA(v18, v17, v14);
  }
  v29 = 0;
LABEL_21:
  v60 = v29;
  v30 = (__m128i *)v29;
  v61 = v29;
  v62 = v29 + v27;
  if ( (const __m128i *)v17 != v26 )
  {
    v31 = (const __m128i *)v17;
    do
    {
      if ( v30 )
      {
        *v30 = _mm_loadu_si128(v31);
        v24 = v31[1].m128i_i64[0];
        v30[1].m128i_i64[0] = v24;
      }
      v31 = (const __m128i *)((char *)v31 + 24);
      v30 = (__m128i *)((char *)v30 + 24);
    }
    while ( v31 != v26 );
    v30 = (__m128i *)(v29 + 8 * (((unsigned __int64)&v31[-2].m128i_u64[1] - v17) >> 3) + 24);
  }
  v32 = v54;
  v33 = v53;
  v61 = (unsigned __int64)v30;
  v34 = (__int64)v30->m128i_i64 - v29;
  if ( v54 - v53 == v34 )
    goto LABEL_47;
  do
  {
LABEL_28:
    v35 = *(_QWORD *)(v32 - 24);
    if ( !*(_BYTE *)(a2 + 28) )
      goto LABEL_80;
    v36 = *(_QWORD **)(a2 + 8);
    v32 = *(unsigned int *)(a2 + 20);
    v34 = (__int64)&v36[v32];
    if ( v36 != (_QWORD *)v34 )
    {
      while ( v35 != *v36 )
      {
        if ( (_QWORD *)v34 == ++v36 )
          goto LABEL_81;
      }
      goto LABEL_33;
    }
LABEL_81:
    if ( (unsigned int)v32 < *(_DWORD *)(a2 + 16) )
    {
      v32 = (unsigned int)(v32 + 1);
      *(_DWORD *)(a2 + 20) = v32;
      *(_QWORD *)v34 = v35;
      ++*(_QWORD *)a2;
    }
    else
    {
LABEL_80:
      sub_C8CC70(a2, v35, v34, v32, v24, v25);
    }
LABEL_33:
    v37 = v54;
    do
    {
      if ( *(_BYTE *)(v37 - 8) )
        goto LABEL_35;
      for ( j = *(_QWORD *)(*(_QWORD *)(v37 - 24) + 16LL); j; j = *(_QWORD *)(j + 8) )
      {
        if ( (unsigned __int8)(**(_BYTE **)(j + 24) - 30) <= 0xAu )
          break;
      }
      *(_QWORD *)(v37 - 16) = j;
      *(_BYTE *)(v37 - 8) = 1;
      while ( 2 )
      {
        if ( j )
        {
          v39 = *(_QWORD *)(j + 8);
          for ( *(_QWORD *)(v37 - 16) = v39; v39; *(_QWORD *)(v37 - 16) = v39 )
          {
            v32 = (unsigned int)**(unsigned __int8 **)(v39 + 24) - 30;
            if ( (unsigned __int8)(**(_BYTE **)(v39 + 24) - 30) <= 0xAu )
              break;
            v39 = *(_QWORD *)(v39 + 8);
          }
          v40 = *(_QWORD *)(*(_QWORD *)(j + 24) + 40LL);
          if ( !v51 )
            goto LABEL_71;
          v41 = v48;
          v32 = v50;
          v39 = (__int64)&v48[v50];
          if ( v48 == (_QWORD *)v39 )
          {
LABEL_43:
            if ( v50 < v49 )
            {
              ++v50;
              *(_QWORD *)v39 = v40;
              ++v47;
LABEL_45:
              v45.m128i_i64[0] = v40;
              v46 = 0;
              sub_2FA90C0((unsigned __int64 *)&v53, &v45);
              v33 = v53;
              v32 = v54;
              goto LABEL_46;
            }
LABEL_71:
            sub_C8CC70((__int64)&v47, v40, v39, v32, v24, v25);
            if ( v43 )
              goto LABEL_45;
          }
          else
          {
            while ( v40 != *v41 )
            {
              if ( (_QWORD *)v39 == ++v41 )
                goto LABEL_43;
            }
          }
LABEL_35:
          j = *(_QWORD *)(v37 - 16);
          continue;
        }
        break;
      }
      v54 -= 24;
      v33 = v53;
      v37 = v54;
    }
    while ( v54 != v53 );
    v32 = v53;
LABEL_46:
    v29 = v60;
    v34 = v61 - v60;
  }
  while ( v32 - v33 != v61 - v60 );
LABEL_47:
  if ( v32 != v33 )
  {
    v34 = v29;
    while ( *(_QWORD *)v33 == *(_QWORD *)v34 )
    {
      v42 = *(_BYTE *)(v33 + 16);
      if ( v42 != *(_BYTE *)(v34 + 16) || v42 && *(_QWORD *)(v33 + 8) != *(_QWORD *)(v34 + 8) )
        break;
      v33 += 24LL;
      v34 += 24;
      if ( v32 == v33 )
        goto LABEL_53;
    }
    goto LABEL_28;
  }
LABEL_53:
  if ( v29 )
    j_j___libc_free_0(v29);
  if ( !v58 )
    _libc_free(v57);
  if ( v53 )
    j_j___libc_free_0(v53);
  if ( !v51 )
    _libc_free((unsigned __int64)v48);
  if ( v70 )
    j_j___libc_free_0((unsigned __int64)v70);
  if ( !v69 )
    _libc_free(v68);
  if ( v65 )
    j_j___libc_free_0((unsigned __int64)v65);
  if ( !v64 )
    _libc_free(v63[1]);
}

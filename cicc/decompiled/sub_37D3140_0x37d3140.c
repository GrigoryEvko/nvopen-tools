// Function: sub_37D3140
// Address: 0x37d3140
//
void __fastcall sub_37D3140(__int64 a1, int a2, unsigned __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r15
  unsigned int v7; // edx
  __int64 v8; // r9
  int v9; // edi
  __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 **v12; // rbx
  int v13; // eax
  __int64 v14; // rsi
  int v15; // ecx
  int v16; // edx
  unsigned int v17; // eax
  int v18; // edi
  __int64 *v19; // r13
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  int v27; // edx
  __int64 v28; // rdi
  __int64 v29; // rax
  unsigned int v30; // edx
  int v31; // esi
  __int64 v32; // rdx
  const __m128i *v33; // rbx
  __int64 v34; // r15
  __int64 v35; // rax
  int v36; // esi
  unsigned __int64 v37; // r11
  unsigned __int64 v38; // rdx
  unsigned int v39; // edx
  __int64 v40; // r10
  __int64 v41; // r12
  int v42; // edx
  const __m128i *v43; // r12
  char *v44; // rdx
  __m128i *v45; // rax
  __int64 v46; // rdx
  __m128i v47; // xmm1
  __m128i v48; // xmm0
  __int64 *v49; // rdx
  int v50; // esi
  __int64 v51; // rdi
  unsigned __int64 v52; // r10
  unsigned __int64 v53; // rax
  unsigned int v54; // eax
  __int64 *v55; // rbx
  __int64 v56; // rax
  __int64 *v57; // rax
  char v58; // cl
  __int64 v59; // rsi
  __int64 v60; // rax
  unsigned int v61; // esi
  char *v62; // rax
  char v63; // cl
  __int64 v64; // rsi
  unsigned int v65; // esi
  __int64 v66; // r10
  __int64 v67; // r12
  int v68; // r10d
  __int64 v69; // rdx
  unsigned __int64 v70; // rdx
  unsigned __int64 v71; // r12
  unsigned __int64 v72; // rbx
  __int64 v73; // rax
  unsigned __int64 *v74; // rax
  int v75; // ebx
  int v76; // r11d
  int i; // r9d
  int v78; // r10d
  int v79; // r10d
  unsigned __int64 v81; // [rsp+10h] [rbp-150h]
  __int64 v82; // [rsp+18h] [rbp-148h]
  __int64 v83; // [rsp+20h] [rbp-140h]
  unsigned __int8 v84; // [rsp+20h] [rbp-140h]
  int *v85; // [rsp+28h] [rbp-138h]
  __int64 **j; // [rsp+38h] [rbp-128h]
  __int64 v87; // [rsp+38h] [rbp-128h]
  __int64 v88; // [rsp+38h] [rbp-128h]
  _OWORD v89[2]; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v90; // [rsp+90h] [rbp-D0h]
  char v91; // [rsp+98h] [rbp-C8h]
  char *v92; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v93; // [rsp+A8h] [rbp-B8h]
  _BYTE v94[48]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v95; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v96; // [rsp+E8h] [rbp-78h]
  __int64 *v97; // [rsp+F0h] [rbp-70h] BYREF
  unsigned int v98; // [rsp+F8h] [rbp-68h]
  char v99; // [rsp+130h] [rbp-30h] BYREF

  v3 = *(unsigned int *)(a1 + 3576);
  v4 = *(_QWORD *)(a1 + 3560);
  if ( !(_DWORD)v3 )
    return;
  v5 = (unsigned int)(v3 - 1);
  v6 = a1;
  v7 = v5 & (37 * a2);
  v8 = 7LL * v7;
  v85 = (int *)(v4 + 112LL * v7);
  v9 = *v85;
  if ( *v85 != a2 )
  {
    for ( i = 1; ; i = v78 )
    {
      if ( v9 == -1 )
        return;
      v78 = i + 1;
      v7 = v5 & (i + v7);
      v8 = v7;
      v85 = (int *)(v4 + 112LL * v7);
      v9 = *v85;
      if ( *v85 == a2 )
        break;
    }
  }
  if ( v85 == (int *)(v4 + 112 * v3) )
    return;
  v95 = 0;
  v96 = 1;
  v10 = unk_5051170;
  v11 = (__int64 *)&v97;
  do
  {
    *v11 = v10;
    v11 += 2;
  }
  while ( v11 != (__int64 *)&v99 );
  v12 = (__int64 **)*((_QWORD *)v85 + 1);
  for ( j = &v12[11 * (unsigned int)v85[4]]; j != v12; v12 += 11 )
  {
    v13 = *(_DWORD *)(v6 + 3608);
    v14 = *(_QWORD *)(v6 + 3592);
    if ( v13 )
    {
      v15 = *((_DWORD *)v12 + 16);
      v16 = v13 - 1;
      v17 = (v13 - 1) & (37 * v15);
      v18 = *(_DWORD *)(v14 + 4LL * v17);
      if ( v15 == v18 )
      {
LABEL_9:
        v19 = *v12;
        v20 = (__int64)&(*v12)[6 * *((unsigned int *)v12 + 2)];
        if ( (__int64 *)v20 != *v12 )
        {
          do
          {
            if ( !*((_BYTE *)v19 + 40) )
            {
              v21 = *v19;
              DWORD2(v89[0]) = 0;
              *(_QWORD *)&v89[0] = v21;
              sub_37D2FB0((__int64)&v92, (__int64)&v95, (__int64 *)v89, (_DWORD *)v89 + 2);
            }
            v19 += 6;
          }
          while ( (__int64 *)v20 != v19 );
        }
      }
      else
      {
        v5 = 1;
        while ( v18 != -1 )
        {
          v8 = (unsigned int)(v5 + 1);
          v17 = v16 & (v5 + v17);
          v18 = *(_DWORD *)(v14 + 4LL * v17);
          if ( v15 == v18 )
            goto LABEL_9;
          v5 = (unsigned int)v8;
        }
      }
    }
  }
  if ( !((unsigned int)v96 >> 1) )
    goto LABEL_46;
  v22 = *(_QWORD *)(v6 + 16);
  v23 = 0;
  v87 = *(unsigned int *)(v22 + 40);
  if ( (_DWORD)v87 )
  {
    do
    {
      v57 = (__int64 *)(*(_QWORD *)(v22 + 32) + 8 * v23);
      v58 = v96 & 1;
      if ( (v96 & 1) != 0 )
      {
        v49 = (__int64 *)&v97;
        v50 = 3;
      }
      else
      {
        v59 = v98;
        v49 = v97;
        if ( !v98 )
          goto LABEL_59;
        v50 = v98 - 1;
      }
      v8 = unk_5051170;
      v51 = *v57;
      v52 = HIDWORD(*v57);
      v53 = 0x9DDFEA08EB382D69LL * (v52 ^ (((8 * *v57) & 0x7FFFFFFF8LL) + 12995744));
      v54 = v50
          & (-348639895
           * (((unsigned int)((0x9DDFEA08EB382D69LL * (v53 ^ v52 ^ (v53 >> 47))) >> 32) >> 15)
            ^ (-348639895 * (v53 ^ v52 ^ (v53 >> 47)))));
      v55 = &v49[2 * v54];
      v5 = *v55;
      if ( *v55 == v51 )
        goto LABEL_37;
      v75 = 1;
      while ( unk_5051170 != v5 )
      {
        v79 = v75 + 1;
        v54 = v50 & (v75 + v54);
        v55 = &v49[2 * v54];
        v5 = *v55;
        if ( v51 == *v55 )
          goto LABEL_37;
        v75 = v79;
      }
      if ( v58 )
      {
        v64 = 8;
        goto LABEL_60;
      }
      v59 = v98;
LABEL_59:
      v64 = 2 * v59;
LABEL_60:
      v55 = &v49[v64];
LABEL_37:
      v56 = 8;
      if ( !v58 )
        v56 = 2LL * v98;
      if ( v55 == &v49[v56] )
        goto LABEL_40;
      v5 = *((unsigned __int8 *)v55 + 11);
      if ( (unsigned __int8)v5 > 2u )
        goto LABEL_40;
      v60 = *(_QWORD *)(v6 + 16);
      v61 = *(_DWORD *)(*(_QWORD *)(v60 + 88) + 4 * v23);
      if ( v61 >= *(_DWORD *)(v60 + 284) )
      {
        v63 = 3;
LABEL_57:
        *((_DWORD *)v55 + 2) = v23 & 0xFFFFFF | v55[1] & 0xFF000000;
        *((_BYTE *)v55 + 11) = v63;
        goto LABEL_40;
      }
      v84 = *((_BYTE *)v55 + 11);
      if ( (_BYTE)v5 != 2 )
      {
        sub_37B9A30(&v92, v61, *(_QWORD **)(v6 + 3616), 1);
        v62 = v92;
        v5 = v84;
        if ( (char *)v93 == v92 )
        {
LABEL_99:
          v63 = 1;
          if ( v84 )
            goto LABEL_40;
        }
        else
        {
          v8 = *(_QWORD *)(v6 + 3624);
          while ( (*(_QWORD *)(*(_QWORD *)v8 + 8 * ((unsigned __int64)*(unsigned __int16 *)v62 >> 6))
                 & (1LL << *(_WORD *)v62)) == 0 )
          {
            v62 += 2;
            v92 = v62;
            if ( (char *)v93 == v62 )
              goto LABEL_99;
          }
          v63 = 2;
        }
        goto LABEL_57;
      }
LABEL_40:
      ++v23;
    }
    while ( v87 != v23 );
  }
  v24 = *((_QWORD *)v85 + 1);
  v25 = (unsigned int)v85[4];
  v26 = 5 * v25;
  v82 = v24 + 88 * v25;
  if ( v82 == v24 )
    goto LABEL_45;
  v88 = *((_QWORD *)v85 + 1);
  v83 = v6;
  do
  {
    v27 = *(_DWORD *)(v83 + 3608);
    v28 = *(_QWORD *)(v83 + 3592);
    if ( !v27 )
      goto LABEL_21;
    v26 = (unsigned int)(v27 - 1);
    v29 = *(unsigned int *)(v88 + 64);
    v30 = v26 & (37 * v29);
    v31 = *(_DWORD *)(v28 + 4LL * v30);
    if ( v31 == (_DWORD)v29 )
    {
LABEL_24:
      v92 = v94;
      v93 = 0x100000000LL;
      v32 = *(unsigned int *)(v88 + 8);
      v33 = *(const __m128i **)v88;
      v34 = *(_QWORD *)v88 + 48 * v32;
      if ( v34 == *(_QWORD *)v88 )
      {
        if ( (_DWORD)v32 )
          goto LABEL_21;
        goto LABEL_79;
      }
      v35 = 0;
      while ( v33[2].m128i_i8[8] )
      {
        v46 = v33[2].m128i_i64[0];
        v47 = _mm_loadu_si128(v33);
        v5 = v35 + 1;
        v43 = (const __m128i *)v89;
        v48 = _mm_loadu_si128(v33 + 1);
        v26 = HIDWORD(v93);
        v91 = 1;
        v90 = v46;
        v44 = v92;
        v89[0] = v47;
        v89[1] = v48;
        if ( v35 + 1 > (unsigned __int64)HIDWORD(v93) )
        {
          if ( v92 <= (char *)v89 && v89 < (_OWORD *)&v92[48 * v35] )
            goto LABEL_69;
          goto LABEL_74;
        }
LABEL_31:
        v45 = (__m128i *)&v44[48 * v35];
        *v45 = _mm_loadu_si128(v43);
        v45[1] = _mm_loadu_si128(v43 + 1);
        v45[2] = _mm_loadu_si128(v43 + 2);
        v35 = (unsigned int)(v93 + 1);
        LODWORD(v93) = v93 + 1;
        v33 += 3;
        if ( (const __m128i *)v34 == v33 )
          goto LABEL_18;
      }
      v26 = v96 & 1;
      if ( (v96 & 1) != 0 )
      {
        v5 = (__int64)&v97;
        v36 = 3;
      }
      else
      {
        v65 = v98;
        v5 = (__int64)v97;
        if ( !v98 )
          goto LABEL_65;
        v36 = v98 - 1;
      }
      v8 = unk_5051170;
      v37 = HIDWORD(v33->m128i_i64[0]);
      v38 = 0x9DDFEA08EB382D69LL * (v37 ^ (((8 * v33->m128i_i64[0]) & 0x7FFFFFFF8LL) + 12995744));
      v39 = v36
          & (-348639895
           * (((unsigned int)((0x9DDFEA08EB382D69LL * (v37 ^ v38 ^ (v38 >> 47))) >> 32) >> 15)
            ^ (-348639895 * (v37 ^ v38 ^ (v38 >> 47)))));
      v40 = v5 + 16LL * v39;
      v41 = *(_QWORD *)v40;
      if ( v33->m128i_i64[0] == *(_QWORD *)v40 )
        goto LABEL_29;
      v68 = 1;
      while ( unk_5051170 != v41 )
      {
        v76 = v68 + 1;
        v39 = v36 & (v68 + v39);
        v40 = v5 + 16LL * v39;
        v41 = *(_QWORD *)v40;
        if ( v33->m128i_i64[0] == *(_QWORD *)v40 )
          goto LABEL_29;
        v68 = v76;
      }
      if ( (_BYTE)v26 )
      {
        v66 = 64;
        goto LABEL_66;
      }
      v65 = v98;
LABEL_65:
      v66 = 16LL * v65;
LABEL_66:
      v40 = v5 + v66;
LABEL_29:
      if ( *(_BYTE *)(v40 + 11) )
      {
        v42 = *(_DWORD *)(v40 + 8);
        v26 = HIDWORD(v93);
        v5 = v35 + 1;
        v43 = (const __m128i *)v89;
        v91 = 0;
        LODWORD(v89[0]) = v42 & 0xFFFFFF;
        v44 = v92;
        if ( v35 + 1 > (unsigned __int64)HIDWORD(v93) )
        {
          if ( v92 <= (char *)v89 && v89 < (_OWORD *)&v92[48 * v35] )
          {
LABEL_69:
            v67 = (char *)v89 - v92;
            sub_C8D5F0((__int64)&v92, v94, v5, 0x30u, v5, v8);
            v44 = v92;
            v35 = (unsigned int)v93;
            v43 = (const __m128i *)&v92[v67];
            goto LABEL_31;
          }
LABEL_74:
          v43 = (const __m128i *)v89;
          sub_C8D5F0((__int64)&v92, v94, v5, 0x30u, v5, v8);
          v44 = v92;
          v35 = (unsigned int)v93;
        }
        goto LABEL_31;
      }
LABEL_18:
      if ( *(_DWORD *)(v88 + 8) != (_DWORD)v35 )
      {
LABEL_19:
        if ( v92 != v94 )
          _libc_free((unsigned __int64)v92);
        goto LABEL_21;
      }
      v29 = *(unsigned int *)(v88 + 64);
LABEL_79:
      v69 = *(_QWORD *)(*(_QWORD *)(v83 + 32) + 32LL) + 48 * v29;
      sub_37BA660(*(_QWORD **)(v83 + 16), (__int64)&v92, v69, *(_QWORD *)(v69 + 40), v88 + 72);
      v26 = *(unsigned int *)(v83 + 3484);
      v71 = v70;
      v72 = *(unsigned int *)(v88 + 64) | v81 & 0xFFFFFFFF00000000LL;
      v73 = *(unsigned int *)(v83 + 3480);
      v81 = v72;
      if ( v73 + 1 > v26 )
      {
        sub_C8D5F0(v83 + 3472, (const void *)(v83 + 3488), v73 + 1, 0x10u, v5, v8);
        v73 = *(unsigned int *)(v83 + 3480);
      }
      v74 = (unsigned __int64 *)(*(_QWORD *)(v83 + 3472) + 16 * v73);
      *v74 = v72;
      v74[1] = v71;
      ++*(_DWORD *)(v83 + 3480);
      goto LABEL_19;
    }
    v5 = 1;
    while ( v31 != -1 )
    {
      v8 = (unsigned int)(v5 + 1);
      v30 = v26 & (v5 + v30);
      v31 = *(_DWORD *)(v28 + 4LL * v30);
      if ( (_DWORD)v29 == v31 )
        goto LABEL_24;
      v5 = (unsigned int)v8;
    }
LABEL_21:
    v88 += 88;
  }
  while ( v82 != v88 );
  v6 = v83;
LABEL_45:
  sub_37C43E0(v6, a3, 0, v26, v5, v8);
LABEL_46:
  if ( (v96 & 1) == 0 )
    sub_C7D6A0((__int64)v97, 16LL * v98, 8);
}

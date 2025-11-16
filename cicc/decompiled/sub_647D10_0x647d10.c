// Function: sub_647D10
// Address: 0x647d10
//
size_t __fastcall sub_647D10(
        const __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        int a6,
        int a7,
        _QWORD *a8,
        __int64 *a9)
{
  __int64 v11; // r13
  int v12; // r15d
  __int8 v13; // al
  _BOOL8 v14; // r14
  __int64 v15; // rax
  _BYTE *v16; // r11
  size_t v17; // r8
  __int64 v18; // rax
  __int64 v20; // rbx
  __int64 v21; // r10
  __int64 v22; // rsi
  __int64 v23; // r10
  char v24; // al
  __int64 i; // rax
  __int64 v26; // rdi
  __int64 v27; // rcx
  _BOOL8 v28; // r8
  __int64 v29; // r8
  _QWORD *v30; // r11
  _BYTE *v31; // r10
  char v32; // al
  int v33; // r14d
  char v34; // al
  __int64 v35; // rax
  __int64 v36; // rbx
  __int64 v37; // rdx
  char v38; // al
  __int64 v39; // rax
  int v40; // eax
  char v41; // al
  const char *v42; // rdi
  __int64 v43; // rdx
  __int64 j; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rcx
  __m128i *v49; // rdi
  const __m128i *v50; // rsi
  char v51; // al
  const char *v52; // rsi
  int v53; // eax
  __int64 v54; // rsi
  unsigned __int8 v55; // al
  int v56; // eax
  bool v57; // r14
  __int64 v58; // rsi
  int v59; // eax
  __int64 v60; // r14
  __int64 v61; // rdi
  char *v62; // rdx
  char *v63; // rcx
  __m128i v64; // xmm5
  __m128i v65; // xmm6
  __m128i v66; // xmm7
  __int64 v67; // rcx
  __m128i *v68; // rdi
  const __m128i *v69; // rsi
  __int64 v70; // rcx
  __m128i *v71; // rdi
  const __m128i *v72; // rsi
  __int64 v73; // rax
  __int64 v74; // [rsp+10h] [rbp-110h]
  _BYTE *v75; // [rsp+18h] [rbp-108h]
  __int64 v76; // [rsp+18h] [rbp-108h]
  __int64 v77; // [rsp+20h] [rbp-100h]
  _BYTE *v78; // [rsp+20h] [rbp-100h]
  __int64 v79; // [rsp+20h] [rbp-100h]
  _QWORD *v80; // [rsp+20h] [rbp-100h]
  _BYTE *v81; // [rsp+20h] [rbp-100h]
  _BYTE *srca; // [rsp+28h] [rbp-F8h]
  _BYTE *src; // [rsp+28h] [rbp-F8h]
  _QWORD *srcb; // [rsp+28h] [rbp-F8h]
  char *srcd; // [rsp+28h] [rbp-F8h]
  _QWORD *srcc; // [rsp+28h] [rbp-F8h]
  size_t na; // [rsp+30h] [rbp-F0h]
  size_t n; // [rsp+30h] [rbp-F0h]
  size_t nb; // [rsp+30h] [rbp-F0h]
  __int64 v91; // [rsp+40h] [rbp-E0h]
  _BOOL4 v92; // [rsp+48h] [rbp-D8h]
  unsigned __int8 v95; // [rsp+58h] [rbp-C8h]
  __int64 v96; // [rsp+58h] [rbp-C8h]
  size_t v97; // [rsp+58h] [rbp-C8h]
  unsigned __int8 v98; // [rsp+60h] [rbp-C0h]
  _BYTE *v99; // [rsp+60h] [rbp-C0h]
  size_t v100; // [rsp+60h] [rbp-C0h]
  size_t v101; // [rsp+60h] [rbp-C0h]
  size_t v102; // [rsp+60h] [rbp-C0h]
  size_t v104; // [rsp+68h] [rbp-B8h]
  __m128i v105; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v106; // [rsp+80h] [rbp-A0h]
  __m128i v107; // [rsp+90h] [rbp-90h]
  __m128i v108; // [rsp+A0h] [rbp-80h]
  _QWORD v109[2]; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v110; // [rsp+C0h] [rbp-60h]
  __m128i v111; // [rsp+D0h] [rbp-50h]
  __m128i v112; // [rsp+E0h] [rbp-40h]

  v11 = *(_QWORD *)(a4 + 48);
  v95 = *(_BYTE *)(a4 + 84);
  if ( v11 )
  {
    v98 = 15;
    v12 = 1;
    v92 = (*(_BYTE *)(v11 + 64) & 0x40) != 0;
  }
  else
  {
    v92 = 0;
    v12 = 0;
    v98 = 14;
  }
  v13 = a1[1].m128i_i8[1];
  LODWORD(v14) = (v13 & 0x20) != 0;
  if ( a7 | v14 )
  {
    v105 = _mm_loadu_si128(a1);
    v106 = _mm_loadu_si128(a1 + 1);
    v107 = _mm_loadu_si128(a1 + 2);
    v108 = _mm_loadu_si128(a1 + 3);
    if ( (v13 & 0x40) == 0 )
    {
      v106.m128i_i8[0] &= ~0x80u;
      v106.m128i_i64[1] = 0;
      if ( ((unsigned __int8)v12 & ((v13 & 0x20) == 0)) == 0 )
        goto LABEL_6;
      goto LABEL_12;
    }
    if ( ((unsigned __int8)v12 & ((v13 & 0x20) == 0)) != 0 )
    {
LABEL_12:
      if ( dword_4F077C4 != 2 || v95 != 3 )
        goto LABEL_6;
LABEL_14:
      if ( (*(_BYTE *)(v11 + 64) & 0x20) == 0 && *(_QWORD *)(a1->m128i_i64[0] + 40) )
      {
        v91 = a4;
        v20 = *(_QWORD *)(a1->m128i_i64[0] + 40);
        while ( 1 )
        {
          if ( *(_BYTE *)(v20 + 80) == 15 && *(_DWORD *)(v20 + 40) == unk_4F066A8 )
          {
            v21 = *(_QWORD *)(*(_QWORD *)(v20 + 88) + 8LL);
            if ( (*(_BYTE *)(v21 + 88) & 0x70) == 0x30 )
            {
              v22 = *(_QWORD *)(v21 + 152);
              v96 = *(_QWORD *)(*(_QWORD *)(v20 + 88) + 8LL);
              if ( v22 != a3 && !(unsigned int)sub_8DED30(a3, v22, 1314824) )
              {
                v23 = v96;
                if ( !(_DWORD)qword_4F077B4
                  || (v24 = *(_BYTE *)(v20 + 80), v24 != 17)
                  && (*(_BYTE *)(v91 + 65) & 8) == 0
                  && (v24 != 11 || (v73 = sub_736C60(20, *(_QWORD *)(*(_QWORD *)(v20 + 88) + 104LL)), v23 = v96, !v73)) )
                {
                  if ( *(_BYTE *)(v23 + 174) || !*(_WORD *)(v23 + 176) )
                    break;
                  for ( i = *(_QWORD *)(v23 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
                    ;
                  if ( (*(_BYTE *)(*(_QWORD *)(i + 168) + 16LL) & 1) == 0 )
                    break;
                }
              }
            }
          }
          v20 = *(_QWORD *)(v20 + 8);
          if ( !v20 )
            goto LABEL_6;
        }
        if ( dword_4F077BC )
        {
          LODWORD(v14) = 1;
          v26 = 8;
          if ( dword_4F04C64 == dword_4F04C34 )
          {
            LODWORD(v14) = *(_QWORD *)(v20 + 64) == *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 224);
            v26 = 3 * v14 + 5;
          }
        }
        else
        {
          LODWORD(v14) = 1;
          v26 = 8;
        }
        sub_6853B0(v26, 338, &a1->m128i_u64[1], v20);
      }
    }
LABEL_6:
    v15 = sub_8862B0(v98, &v105);
    v16 = *(_BYTE **)(v15 + 88);
    v17 = v15;
    if ( v98 == 15
      && ((v16[16] = v92, v51 = *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 4), ((v51 - 15) & 0xFD) == 0)
       || v51 == 2) )
    {
      v97 = v17;
      v99 = v16;
      v18 = sub_73F2D0(a3);
      v17 = v97;
      v16 = v99;
    }
    else
    {
      v18 = a3;
    }
    *(_QWORD *)v16 = v18;
    goto LABEL_9;
  }
  v27 = *(_QWORD *)(a2 + 400);
  v28 = (*(_BYTE *)(a4 + 65) & 8) != 0;
  if ( v12 )
  {
    v29 = sub_879D20(a1, v95, a3, v27, v28, &v105);
    if ( !v29 )
      goto LABEL_12;
  }
  else
  {
    v29 = sub_879D20(a1, v95, 0, v27, v28, &v105);
    if ( !v29 )
      goto LABEL_68;
  }
  v30 = *(_QWORD **)(v29 + 88);
  v31 = (_BYTE *)v30[1];
  if ( dword_4F077C4 == 2 && dword_4F077BC && !(_DWORD)qword_4F077B4 && dword_4F04C58 == -1 && v95 == 3 )
  {
    if ( v12 )
    {
      if ( (v31[193] & 0x20) == 0 )
      {
        v48 = 16;
        v49 = &v105;
        v50 = a1;
        while ( v48 )
        {
          v49->m128i_i32[0] = v50->m128i_i32[0];
          v50 = (const __m128i *)((char *)v50 + 4);
          v49 = (__m128i *)((char *)v49 + 4);
          --v48;
        }
        if ( (v106.m128i_i8[1] & 0x40) == 0 )
        {
          v106.m128i_i8[0] &= ~0x80u;
          v106.m128i_i64[1] = 0;
        }
        goto LABEL_14;
      }
LABEL_155:
      if ( (*(_BYTE *)(a4 + 64) & 8) == 0 )
      {
        v67 = 16;
        v68 = &v105;
        v69 = a1;
        while ( v67 )
        {
          v68->m128i_i32[0] = v69->m128i_i32[0];
          v69 = (const __m128i *)((char *)v69 + 4);
          v68 = (__m128i *)((char *)v68 + 4);
          --v67;
        }
        if ( (v106.m128i_i8[1] & 0x40) == 0 )
        {
          v106.m128i_i8[0] &= ~0x80u;
          v106.m128i_i64[1] = 0;
        }
        if ( v12 )
          goto LABEL_14;
        goto LABEL_68;
      }
      goto LABEL_39;
    }
    if ( v31[136] != 1 )
      goto LABEL_155;
    v70 = 16;
    v71 = &v105;
    v72 = a1;
    while ( v70 )
    {
      v71->m128i_i32[0] = v72->m128i_i32[0];
      v72 = (const __m128i *)((char *)v72 + 4);
      v71 = (__m128i *)((char *)v71 + 4);
      --v70;
    }
    if ( (v106.m128i_i8[1] & 0x40) == 0 )
    {
      v106.m128i_i8[0] &= ~0x80u;
      v106.m128i_i64[1] = 0;
    }
LABEL_68:
    LODWORD(v14) = 0;
    goto LABEL_6;
  }
LABEL_39:
  v32 = *(_BYTE *)(v29 + 80);
  if ( !unk_4D041F8 || !dword_4F077C0 || !*(_QWORD *)v31 )
    goto LABEL_48;
  if ( qword_4F077A8 <= 0x76BFu )
  {
LABEL_129:
    v79 = v29;
    srcb = v30;
    nb = (size_t)v31;
    v56 = sub_85EBD0(*(_QWORD *)v31, v109);
    v29 = v79;
    v31 = (_BYTE *)nb;
    v30 = srcb;
    v57 = v56 != -1;
    v32 = *(_BYTE *)(v79 + 80);
    LOBYTE(v33) = 3 * v57 + 5;
    goto LABEL_49;
  }
  if ( v32 != v98 || !v12 )
  {
LABEL_48:
    LOBYTE(v33) = 8;
    goto LABEL_49;
  }
  if ( !v92 && !*((_BYTE *)v30 + 16) )
  {
    v33 = a6 == 0 ? 8 : 3;
LABEL_70:
    v38 = *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 4);
    if ( ((v38 - 15) & 0xFD) == 0 || (v37 = a3, v38 == 2) )
    {
      v77 = v29;
      srca = v31;
      na = (size_t)v30;
      v39 = sub_73F2D0(a3);
      v29 = v77;
      v31 = srca;
      v30 = (_QWORD *)na;
      v37 = v39;
    }
    goto LABEL_73;
  }
  if ( qword_4F077A8 > 0x9C3Fu )
    goto LABEL_129;
  v60 = a1->m128i_i64[0];
  v61 = *(_QWORD *)(a1->m128i_i64[0] + 16);
  if ( (unsigned __int64)(v61 + 11) > unk_4F06C48 )
  {
    v76 = v29;
    v81 = (_BYTE *)v30[1];
    srcc = *(_QWORD **)(v29 + 88);
    sub_729510();
    v29 = v76;
    v31 = v81;
    v30 = srcc;
  }
  v74 = v29;
  v62 = (char *)qword_4F06C50;
  v75 = v31;
  v80 = v30;
  *qword_4F06C50 = 0x69746C6975625F5FLL;
  v63 = (char *)qword_4F06C50;
  strcpy(v62 + 8, "n_");
  srcd = v63;
  strcpy(v63 + 10, *(const char **)(v60 + 8));
  v64 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v65 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v66 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v109[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v110 = v64;
  v111 = v65;
  v112 = v66;
  v109[1] = unk_4F077C8;
  sub_878540(srcd, v61 + 10);
  v30 = v80;
  v31 = v75;
  v29 = v74;
  if ( v109[0] )
  {
    if ( (*(_BYTE *)(v109[0] + 73LL) & 0x20) != 0 )
      goto LABEL_129;
  }
  v32 = *(_BYTE *)(v74 + 80);
  LOBYTE(v33) = 8;
LABEL_49:
  if ( a6 )
    LOBYTE(v33) = 3;
  if ( v98 != v32 )
  {
    if ( dword_4F077BC && (*(_BYTE *)(a4 + 64) & 0x10) != 0 )
    {
      v34 = v32 == 15 ? (v31[88] >> 4) & 7 : *(_BYTE *)(a4 + 84);
      if ( v34 == 2 )
        LOBYTE(v33) = 5;
    }
    if ( !a6 )
    {
      v47 = (unsigned __int8)v33;
      LODWORD(v14) = 1;
      sub_6853B0(v47, 159, &a1->m128i_u64[1], v29);
      goto LABEL_6;
    }
    goto LABEL_59;
  }
  v37 = a3;
  if ( v12 )
    goto LABEL_70;
LABEL_73:
  v78 = v31;
  src = v30;
  n = v29;
  v40 = sub_6476B0(v29, &a1->m128i_i64[1], v37, v33);
  v17 = n;
  v16 = src;
  v14 = v40 == 0;
  if ( !v92 && v98 == 15 )
    src[16] = 0;
  if ( !(v14 | a5) )
  {
    v41 = v78[89];
    v42 = 0;
    if ( (v41 & 0x40) == 0 )
    {
      if ( (v41 & 8) != 0 )
        v42 = (const char *)*((_QWORD *)v78 + 3);
      else
        v42 = (const char *)*((_QWORD *)v78 + 1);
    }
    if ( dword_4F077C4 != 2 )
      goto LABEL_60;
    if ( v12 )
      goto LABEL_82;
    if ( ((v78[88] >> 4) & 7) != v95 && ((v78[88] & 0x70) == 0x30 || v95 == 3) )
    {
      if ( dword_4D04964 )
      {
        v55 = byte_4F07472[0];
        if ( byte_4F07472[0] == 3 )
        {
LABEL_61:
          if ( *a8 )
            return v17;
          v35 = *(_QWORD *)(*(_QWORD *)(v17 + 88) + 8LL);
          *a8 = v35;
          if ( !v35 )
            return v17;
          v36 = *(_QWORD *)(v35 + 120);
          if ( dword_4F04C58 == -1 || !dword_4F077C0 )
            *(_QWORD *)(v35 + 120) = a3;
          else
            *(_QWORD *)(v35 + 120) = *(_QWORD *)v16;
          goto LABEL_93;
        }
        v54 = 1556;
      }
      else
      {
        v54 = 1556;
        v55 = 5;
      }
    }
    else
    {
      v52 = *(const char **)(a1->m128i_i64[0] + 8);
      if ( v42 == v52 )
        goto LABEL_61;
      v53 = strcmp(v42, v52);
      v16 = src;
      v17 = n;
      if ( !v53 )
        goto LABEL_61;
      v54 = 160;
      v55 = 7;
    }
    if ( !a6 )
    {
      sub_6853B0(v55, v54, &a1->m128i_u64[1], v17);
      LODWORD(v14) = 1;
      goto LABEL_6;
    }
LABEL_59:
    LODWORD(v14) = 1;
    goto LABEL_6;
  }
LABEL_9:
  if ( v14 )
    return v17;
LABEL_60:
  if ( !v12 )
    goto LABEL_61;
LABEL_82:
  if ( *a9 )
    return v17;
  v43 = *(_QWORD *)(*(_QWORD *)(v17 + 88) + 8LL);
  *a9 = v43;
  if ( !v43 )
    return v17;
  if ( dword_4F077C4 == 2 )
  {
    v58 = *(_QWORD *)v43;
    if ( (*(_BYTE *)(v11 + 64) & 4) != 0
      && ((*(_BYTE *)(v43 + 193) & 0x20) != 0 || *(_DWORD *)(v43 + 160) || *(_QWORD *)(v43 + 344)) )
    {
      v100 = v17;
      sub_685920(&a1->m128i_u64[1], v58, 8);
      v17 = v100;
      *a9 = 0;
      *(_QWORD *)(*(_QWORD *)(v100 + 88) + 8LL) = 0;
    }
    else
    {
      v102 = v17;
      sub_6464A0(a3, v58, (unsigned int *)(v11 + 24), 1u);
      v17 = v102;
    }
    v43 = *a9;
    if ( !*a9 )
      return v17;
  }
  v36 = *(_QWORD *)(v43 + 152);
  for ( j = v36; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  if ( !*(_QWORD *)(*(_QWORD *)(j + 168) + 8LL) || dword_4F04C5C != dword_4F04C34 )
  {
    if ( (*(_BYTE *)(a2 + 125) & 8) != 0 && (*(_BYTE *)(v43 + 207) & 0x20) != 0 )
      *(_QWORD *)(a3 + 160) = *(_QWORD *)(v36 + 160);
    *(_QWORD *)(*a9 + 152) = a3;
  }
LABEL_93:
  if ( !*(_QWORD *)(a2 + 296) )
    *(_QWORD *)(a2 + 296) = v36;
  if ( a3 != v36 )
  {
    if ( !a3 || !v36 || !dword_4F07588 || (v45 = *(_QWORD *)(a3 + 32), *(_QWORD *)(v36 + 32) != v45) || !v45 )
    {
      if ( dword_4F077C4 != 1 || (v101 = v17, v59 = sub_8D97D0(a3, v36, 0, a3, v17), v17 = v101, v59) )
      {
        if ( dword_4F04C5C != dword_4F04C34 )
        {
          v104 = v17;
          v46 = sub_87E220();
          v17 = v104;
          *(_QWORD *)(v46 + 8) = v36;
          *(_DWORD *)(v46 + 16) = v12;
          if ( v12 )
            *(_QWORD *)(v46 + 24) = *a9;
          else
            *(_QWORD *)(v46 + 24) = *a8;
        }
      }
    }
  }
  return v17;
}

// Function: sub_8C4010
// Address: 0x8c4010
//
_QWORD *__fastcall sub_8C4010(__int64 a1)
{
  char v2; // di
  bool v3; // zf
  char v4; // di
  _QWORD *v5; // rax
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // r13
  const __m128i *v9; // r12
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // r14
  __int64 v13; // rbx
  _QWORD *v14; // r14
  __int64 v15; // rax
  _QWORD *v16; // r12
  int v17; // r15d
  _QWORD *v18; // rdx
  __int64 v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rsi
  _QWORD *v22; // rdx
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rcx
  bool v26; // r9
  _BOOL4 v27; // edi
  _QWORD *v28; // r8
  __int64 v29; // rax
  __int64 v30; // rsi
  _QWORD *v31; // rdx
  __int64 v32; // r13
  unsigned int v33; // edx
  __int64 v34; // rax
  _QWORD *v35; // rbx
  _QWORD *v36; // rax
  __int64 v37; // r14
  _QWORD *v38; // r12
  _BOOL4 v39; // edx
  _QWORD *v40; // rax
  __int64 v41; // rbx
  __int64 v42; // r12
  _QWORD *result; // rax
  _QWORD *v44; // rsi
  _QWORD *v45; // rcx
  _QWORD *v46; // rax
  _QWORD *v47; // rdx
  __int64 v48; // r9
  __int64 v49; // r10
  char v50; // dl
  char v51; // r8
  char v52; // cl
  __int64 *v53; // rdx
  char v54; // dl
  __int64 v55; // rax
  __int64 v56; // rax
  _QWORD **v57; // rax
  int v58; // edx
  int v59; // edx
  char v60; // r8
  char v61; // cl
  __int64 *v62; // rax
  __int64 v63; // rsi
  __int64 v64; // rcx
  bool v65; // r8
  _QWORD *v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rsi
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rcx
  _QWORD *v74; // rcx
  __int64 v75; // rcx
  char v76; // dl
  __m128i v77; // xmm7
  __int64 *v78; // rax
  unsigned __int8 v79; // al
  char v80; // si
  __m128i v81; // xmm3
  __int64 v82; // rax
  _QWORD *v83; // rax
  __int64 v84; // [rsp+8h] [rbp-88h]
  __int64 v85; // [rsp+10h] [rbp-80h]
  __int64 v86; // [rsp+18h] [rbp-78h]
  char v87; // [rsp+27h] [rbp-69h]
  __int64 v88; // [rsp+28h] [rbp-68h]
  __int64 v89; // [rsp+28h] [rbp-68h]
  __int64 v90; // [rsp+30h] [rbp-60h]
  bool v91; // [rsp+30h] [rbp-60h]
  __int64 v92; // [rsp+30h] [rbp-60h]
  _BOOL4 v93; // [rsp+38h] [rbp-58h]
  __int64 v94; // [rsp+38h] [rbp-58h]
  __int64 v95; // [rsp+38h] [rbp-58h]
  __int64 v96; // [rsp+40h] [rbp-50h]
  __int64 v97; // [rsp+40h] [rbp-50h]
  int v98; // [rsp+40h] [rbp-50h]
  char v99; // [rsp+40h] [rbp-50h]
  __int64 v100; // [rsp+40h] [rbp-50h]
  char v101; // [rsp+40h] [rbp-50h]
  __int64 v102; // [rsp+40h] [rbp-50h]
  int v103; // [rsp+40h] [rbp-50h]
  __int64 v104; // [rsp+40h] [rbp-50h]
  bool v106; // [rsp+50h] [rbp-40h]
  bool v107; // [rsp+50h] [rbp-40h]
  char v108; // [rsp+51h] [rbp-3Fh]
  bool v109; // [rsp+52h] [rbp-3Eh]
  bool v110; // [rsp+53h] [rbp-3Dh]
  _QWORD *v111; // [rsp+58h] [rbp-38h]

  v2 = *(_BYTE *)(a1 + 28);
  v3 = v2 == 6;
  v108 = v2;
  v4 = *(_BYTE *)(a1 - 8);
  v110 = v3;
  v5 = *(_QWORD **)(a1 - 24);
  v111 = v5;
  v109 = (v4 & 8) != 0;
  if ( (*(_BYTE *)(v5 - 1) & 2) != 0 )
    v111 = (_QWORD *)*(v5 - 3);
  v6 = sub_85EB10((__int64)v111);
  v7 = *(_QWORD *)(a1 + 104);
  if ( v7 )
  {
    v8 = 0;
    if ( v6 )
      v8 = *(_QWORD *)(v6 + 32);
    while ( 1 )
    {
      v9 = *(const __m128i **)(v7 - 24);
      if ( (unsigned __int8)(*(_BYTE *)(v7 + 140) - 9) <= 2u )
      {
        v10 = *(_QWORD *)(v7 + 168);
        if ( v10 )
        {
          if ( *(_QWORD *)(v10 + 152) )
            ((void (*)(void))sub_8C4010)();
        }
      }
      if ( (*(_BYTE *)(v7 - 8) & 8) != 0 )
        break;
      if ( (v4 & 8) != 0 )
      {
        if ( !v8 )
        {
          if ( !v110 || (v34 = v111[13]) == 0 )
          {
            v111[13] = v9;
            goto LABEL_62;
          }
          do
          {
            v8 = v34;
            v34 = *(_QWORD *)(v34 + 112);
          }
          while ( v34 );
        }
        *(_QWORD *)(v8 + 112) = v9;
LABEL_62:
        v9[7].m128i_i64[0] = 0;
        v8 = (__int64)v9;
        if ( v6 )
          *(_QWORD *)(v6 + 32) = v9;
      }
LABEL_14:
      v7 = *(_QWORD *)(v7 + 112);
      if ( !v7 )
        goto LABEL_15;
    }
    v11 = *(_QWORD **)(v7 + 32);
    v12 = v9[-2].m128i_i64[1];
    if ( !v11 || *v11 != v7 || !v11[1] )
    {
      sub_8C3930((__int64)v9, v9[-2].m128i_i64[1]);
      goto LABEL_14;
    }
    if ( v108 != 6 )
    {
      if ( (unsigned __int8)(v9[8].m128i_i8[12] - 9) <= 2u )
      {
        if ( !(unsigned int)sub_8D2490(v9) || (v9[11].m128i_i8[1] & 0x20) != 0 )
          goto LABEL_182;
        goto LABEL_181;
      }
      if ( !(unsigned int)sub_8D23B0(v9) )
      {
LABEL_181:
        v8 = v12;
        sub_736870(v12, -1);
      }
    }
LABEL_182:
    if ( (unsigned __int8)(v9[8].m128i_i8[12] - 9) <= 2u )
    {
      v107 = (*(_BYTE *)(v12 + 88) & 8) != 0;
      v89 = *(_QWORD *)(v12 + 112);
      v92 = v9->m128i_i64[0];
      v95 = *(_QWORD *)(*(_QWORD *)(v12 + 168) + 128LL);
      v101 = (*(_BYTE *)(v12 + 178) & 0x40) != 0;
      sub_8C3930(v12, (__int64)v9);
      *(__m128i *)v12 = _mm_loadu_si128(v9);
      *(__m128i *)(v12 + 16) = _mm_loadu_si128(v9 + 1);
      *(__m128i *)(v12 + 32) = _mm_loadu_si128(v9 + 2);
      *(__m128i *)(v12 + 48) = _mm_loadu_si128(v9 + 3);
      *(__m128i *)(v12 + 64) = _mm_loadu_si128(v9 + 4);
      *(__m128i *)(v12 + 80) = _mm_loadu_si128(v9 + 5);
      v80 = *(_BYTE *)(v12 + 88);
      *(__m128i *)(v12 + 96) = _mm_loadu_si128(v9 + 6);
      *(__m128i *)(v12 + 112) = _mm_loadu_si128(v9 + 7);
      *(__m128i *)(v12 + 128) = _mm_loadu_si128(v9 + 8);
      *(__m128i *)(v12 + 144) = _mm_loadu_si128(v9 + 9);
      *(__m128i *)(v12 + 160) = _mm_loadu_si128(v9 + 10);
      v81 = _mm_loadu_si128(v9 + 11);
      *(_BYTE *)(v12 + 88) = v80 & 0xF7 | (8 * v107);
      v82 = *(_QWORD *)(v12 + 168);
      *(_QWORD *)(v12 + 112) = v89;
      *(__m128i *)(v12 + 176) = v81;
      *(_QWORD *)(v82 + 128) = v95;
      *(_BYTE *)(v12 + 178) = *(_BYTE *)(v12 + 178) & 0xBF | (v101 << 6);
      v75 = v92;
    }
    else
    {
      v91 = (*(_BYTE *)(v12 + 88) & 8) != 0;
      v94 = *(_QWORD *)(v12 + 112);
      v100 = v9->m128i_i64[0];
      sub_8C3930(v12, (__int64)v9);
      v75 = v100;
      *(__m128i *)v12 = _mm_loadu_si128(v9);
      *(__m128i *)(v12 + 16) = _mm_loadu_si128(v9 + 1);
      *(__m128i *)(v12 + 32) = _mm_loadu_si128(v9 + 2);
      *(__m128i *)(v12 + 48) = _mm_loadu_si128(v9 + 3);
      *(__m128i *)(v12 + 64) = _mm_loadu_si128(v9 + 4);
      *(__m128i *)(v12 + 80) = _mm_loadu_si128(v9 + 5);
      v76 = *(_BYTE *)(v12 + 88);
      *(__m128i *)(v12 + 96) = _mm_loadu_si128(v9 + 6);
      *(__m128i *)(v12 + 112) = _mm_loadu_si128(v9 + 7);
      *(__m128i *)(v12 + 128) = _mm_loadu_si128(v9 + 8);
      *(__m128i *)(v12 + 144) = _mm_loadu_si128(v9 + 9);
      *(__m128i *)(v12 + 160) = _mm_loadu_si128(v9 + 10);
      v77 = _mm_loadu_si128(v9 + 11);
      *(_QWORD *)(v12 + 112) = v94;
      *(_BYTE *)(v12 + 88) = v76 & 0xF7 | (8 * v91);
      *(__m128i *)(v12 + 176) = v77;
    }
    v78 = *(__int64 **)(v12 + 32);
    if ( v78 )
      *v78 = v12;
    if ( v75 )
    {
      v79 = *(_BYTE *)(v75 + 80);
      if ( v79 > 5u )
      {
        if ( v79 != 6 )
LABEL_204:
          sub_721090();
      }
      else if ( v79 <= 2u )
      {
        goto LABEL_204;
      }
      *(_QWORD *)(v75 + 88) = v12;
    }
    goto LABEL_14;
  }
LABEL_15:
  v13 = *(_QWORD *)(a1 + 112);
  if ( v13 )
  {
    v14 = 0;
    v15 = v6;
    if ( v6 )
    {
      v14 = *(_QWORD **)(v6 + 40);
      v16 = *(_QWORD **)(v13 - 24);
      if ( (*(_BYTE *)(v13 - 8) & 8) != 0 )
        goto LABEL_46;
      goto LABEL_18;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v16 = *(_QWORD **)(v13 - 24);
        if ( (*(_BYTE *)(v13 - 8) & 8) != 0 )
        {
LABEL_46:
          v31 = *(_QWORD **)(v13 + 32);
          v32 = *(v16 - 3);
          if ( !v31 || *v31 != v13 || !v31[1] )
          {
            v96 = v15;
            *(_BYTE *)(v32 + 169) = (*(_BYTE *)(v32 + 169) | *((_BYTE *)v16 + 169)) & 0x40
                                  | *(_BYTE *)(v32 + 169) & 0xBF;
            sub_8C2B90((__int64)v16, v32);
            v33 = *((_DWORD *)v16 + 38);
            v15 = v96;
            if ( v33 > *(_DWORD *)(v32 + 152) )
              *(_DWORD *)(v32 + 152) = v33;
            goto LABEL_50;
          }
          v17 = 0;
          if ( v108 != 6 && !*((_BYTE *)v16 + 136) )
          {
            v104 = v15;
            sub_735DA0(*(v16 - 3), -1, 0);
            v15 = v104;
            v17 = 1;
            v14 = *(_QWORD **)(v104 + 40);
          }
          if ( !*(_BYTE *)(v32 + 136) )
          {
            v102 = v15;
            sub_735A70(v32);
            v15 = v102;
          }
          v97 = v15;
          sub_734EF0(v32);
          v48 = *v16;
          v49 = *(_QWORD *)(v32 + 112);
          v50 = *(_BYTE *)(v32 + 169);
          v51 = *(_BYTE *)(v32 + 88);
          qmemcpy((void *)v32, v16, 0x110u);
          *(_QWORD *)(v32 + 112) = v49;
          v52 = *(_BYTE *)(v32 + 88);
          *(_BYTE *)(v32 + 169) |= v50 & 0x40;
          v53 = *(__int64 **)(v32 + 32);
          v15 = v97;
          *(_BYTE *)(v32 + 88) = v52 & 0xF7 | v51 & 8;
          if ( v53 )
            *v53 = v32;
          if ( v48 )
          {
            v54 = *(_BYTE *)(v48 + 80);
            if ( v54 != 7 && v54 != 9 )
              goto LABEL_204;
            *(_QWORD *)(v48 + 88) = v32;
            v16 = (_QWORD *)v32;
          }
          else
          {
            v16 = (_QWORD *)v32;
          }
        }
        else
        {
LABEL_18:
          v17 = v109;
        }
        if ( v17 )
          break;
LABEL_50:
        v13 = *(_QWORD *)(v13 + 112);
        if ( !v13 )
        {
LABEL_26:
          v6 = v15;
          goto LABEL_27;
        }
      }
      if ( !v14 )
      {
        if ( !v110 || (v18 = (_QWORD *)v111[14]) == 0 )
        {
          v111[14] = v16;
          goto LABEL_24;
        }
        do
        {
          v14 = v18;
          v18 = (_QWORD *)v18[14];
        }
        while ( v18 );
      }
      v14[14] = v16;
LABEL_24:
      v16[14] = 0;
      v14 = v16;
      if ( !v15 )
        goto LABEL_50;
      *(_QWORD *)(v15 + 40) = v16;
      v13 = *(_QWORD *)(v13 + 112);
      if ( !v13 )
        goto LABEL_26;
    }
  }
LABEL_27:
  v19 = *(_QWORD *)(a1 + 192);
  if ( !v19 || (v4 & 8) == 0 )
  {
    v23 = *(_QWORD *)(a1 + 144);
    if ( v23 )
    {
      v35 = 0;
      if ( !v6 )
        goto LABEL_71;
      goto LABEL_66;
    }
LABEL_80:
    v24 = *(_QWORD *)(a1 + 272);
    if ( !v24 )
      goto LABEL_83;
    v25 = 0;
    if ( v6 )
      goto LABEL_35;
    v26 = v110;
    v27 = v109;
    v28 = v111;
LABEL_41:
    while ( !v27 )
    {
LABEL_40:
      v24 = *(_QWORD *)(v24 + 112);
      if ( !v24 )
        goto LABEL_83;
    }
    v30 = *(_QWORD *)(v24 - 24);
    if ( !v25 )
    {
      if ( !v26 || (v29 = v28[34]) == 0 )
      {
        v28[34] = v30;
        goto LABEL_38;
      }
      do
      {
        v25 = v29;
        v56 = *(_QWORD *)(v29 + 112);
        if ( !v56 )
          break;
        v25 = v56;
        v29 = *(_QWORD *)(v56 + 112);
      }
      while ( v29 );
    }
    *(_QWORD *)(v25 + 112) = v30;
LABEL_38:
    *(_QWORD *)(v30 + 112) = 0;
    v25 = v30;
    if ( v6 )
      *(_QWORD *)(v6 + 104) = v30;
    goto LABEL_40;
  }
  v20 = *(_QWORD **)(v19 - 24);
  if ( (*(_BYTE *)(v20 - 1) & 2) != 0 )
  {
    v21 = v111;
    v20 = (_QWORD *)*(v20 - 3);
    if ( v111[24] )
      goto LABEL_31;
  }
  else
  {
    v21 = v111;
    if ( v111[24] )
    {
LABEL_31:
      **(_QWORD **)(v6 + 64) = v20;
      goto LABEL_32;
    }
  }
  v21[24] = v20;
  do
  {
LABEL_32:
    v22 = v20;
    v20 = (_QWORD *)*v20;
  }
  while ( v20 );
  *(_QWORD *)(v6 + 64) = v22;
  v23 = *(_QWORD *)(a1 + 144);
  if ( v23 )
  {
LABEL_66:
    v35 = *(_QWORD **)(v6 + 48);
LABEL_71:
    while ( 1 )
    {
      v38 = *(_QWORD **)(v23 - 24);
      if ( (*(_BYTE *)(v23 - 8) & 8) == 0 )
        break;
      v36 = *(_QWORD **)(v23 + 32);
      v37 = *(v38 - 3);
      if ( v36 && *v36 == v23 && v36[1] )
      {
        v58 = 0;
        if ( v108 != 6 )
        {
          v58 = *((_DWORD *)v38 + 40);
          if ( v58 )
          {
            sub_736270(*(v38 - 3), -1);
            v35 = *(_QWORD **)(v6 + 48);
            v58 = 1;
          }
        }
        if ( *(_DWORD *)(v37 + 160) )
        {
          v103 = v58;
          v83 = (_QWORD *)sub_72B840(v37);
          sub_734690(v83);
          v58 = v103;
          if ( (*(_BYTE *)(v37 + 195) & 2) == 0 )
          {
            sub_8C3FF0(v37);
            v58 = v103;
          }
        }
        v98 = v58;
        sub_734AA0(v37);
        v59 = v98;
        if ( (*(_BYTE *)(v37 + 194) & 0x40) != 0 )
        {
          v84 = *(_QWORD *)(v37 + 232);
          v85 = 0;
        }
        else
        {
          v85 = *(_QWORD *)(v37 + 232);
          v84 = 0;
        }
        v86 = 0;
        v60 = 0;
        v99 = *(_BYTE *)(v37 + 203);
        if ( (unsigned __int8)(*(_BYTE *)(v37 + 174) - 1) <= 1u )
        {
          v86 = *(_QWORD *)(v37 + 176);
          v60 = (*(_BYTE *)(v37 + 205) >> 2) & 7;
        }
        v87 = v60;
        v106 = (*(_BYTE *)(v37 + 88) & 8) != 0;
        v88 = *(_QWORD *)(v37 + 112);
        v90 = *v38;
        v93 = v59;
        sub_8C2DF0(v37, (__int64)v38);
        qmemcpy((void *)v37, v38, 0x170u);
        v39 = v93;
        v61 = *(_BYTE *)(v37 + 88);
        *(_QWORD *)(v37 + 112) = v88;
        v3 = (*(_BYTE *)(v37 + 194) & 0x40) == 0;
        *(_BYTE *)(v37 + 88) = v61 & 0xF7 | (8 * v106);
        if ( v3 )
          *(_QWORD *)(v37 + 232) = v85;
        else
          *(_QWORD *)(v37 + 232) = v84;
        *(_BYTE *)(v37 + 203) = v99 & 0x84 | *(_BYTE *)(v37 + 203) & 0x7B;
        if ( (unsigned __int8)(*(_BYTE *)(v37 + 174) - 1) <= 1u )
        {
          *(_QWORD *)(v37 + 176) = v86;
          *(_BYTE *)(v37 + 205) = *(_BYTE *)(v37 + 205) & 0xE3 | (4 * (v87 & 7));
        }
        v62 = *(__int64 **)(v37 + 32);
        if ( v62 )
          *v62 = v37;
        if ( v90 )
        {
          if ( (unsigned int)*(unsigned __int8 *)(v90 + 80) - 10 > 1 )
            goto LABEL_204;
          *(_QWORD *)(v90 + 88) = v37;
          v38 = (_QWORD *)v37;
        }
        else
        {
          v38 = (_QWORD *)v37;
        }
LABEL_73:
        if ( !v39 )
          goto LABEL_70;
        if ( !v35 )
        {
          if ( !v110 || (v40 = (_QWORD *)v111[18]) == 0 )
          {
            v111[18] = v38;
            goto LABEL_78;
          }
          do
          {
            v35 = v40;
            v40 = (_QWORD *)v40[14];
          }
          while ( v40 );
        }
        v35[14] = v38;
LABEL_78:
        v38[14] = 0;
        v35 = v38;
        if ( !v6 )
          goto LABEL_70;
        *(_QWORD *)(v6 + 48) = v38;
        v23 = *(_QWORD *)(v23 + 112);
        if ( !v23 )
          goto LABEL_80;
      }
      else
      {
        sub_8C2DF0(*(_QWORD *)(v23 - 24), *(v38 - 3));
LABEL_70:
        v23 = *(_QWORD *)(v23 + 112);
        if ( !v23 )
          goto LABEL_80;
      }
    }
    v39 = v109;
    goto LABEL_73;
  }
  v24 = *(_QWORD *)(a1 + 272);
  if ( v24 )
  {
LABEL_35:
    v25 = *(_QWORD *)(v6 + 104);
    v26 = v110;
    v27 = v109;
    v28 = v111;
    goto LABEL_41;
  }
LABEL_83:
  v41 = *(_QWORD *)(a1 + 168);
  if ( v41 )
  {
    v42 = *(_QWORD *)(v6 + 72);
    do
    {
      if ( (*(_BYTE *)(v41 - 8) & 8) == 0 && v109 )
      {
        v55 = *(_QWORD *)(v41 - 24);
        if ( v42 )
          *(_QWORD *)(v42 + 112) = v55;
        else
          v111[21] = v55;
        *(_QWORD *)(v55 + 112) = 0;
        v42 = v55;
        *(_QWORD *)(v6 + 72) = v55;
      }
      if ( (*(_BYTE *)(v41 + 124) & 1) == 0 )
        sub_8C4010(*(_QWORD *)(v41 + 128));
      v41 = *(_QWORD *)(v41 + 112);
    }
    while ( v41 );
  }
  result = (_QWORD *)a1;
  v44 = *(_QWORD **)(a1 + 232);
  if ( v44 )
  {
    if ( !v109 )
      return result;
    v45 = 0;
    if ( v6 )
      v45 = *(_QWORD **)(v6 + 96);
    while ( 1 )
    {
      v47 = v45;
      v45 = (_QWORD *)*(v44 - 3);
      if ( v47 )
        goto LABEL_102;
      if ( v110 )
      {
        v46 = (_QWORD *)v111[29];
        if ( v46 )
          break;
      }
      v111[29] = v45;
LABEL_96:
      *v45 = 0;
      if ( v6 )
        *(_QWORD *)(v6 + 96) = v45;
      v44 = (_QWORD *)*v44;
      if ( !v44 )
      {
        v63 = *(_QWORD *)(a1 + 152);
        if ( v63 )
        {
          v64 = 0;
          if ( v6 )
          {
LABEL_150:
            v64 = *(_QWORD *)(v6 + 56);
            v65 = v110;
            v66 = v111;
            goto LABEL_156;
          }
LABEL_193:
          v65 = v110;
          v66 = v111;
LABEL_156:
          while ( 2 )
          {
            v68 = v64;
            v64 = *(_QWORD *)(v63 - 24);
            if ( !v68 )
            {
              if ( !v65 || (v67 = v66[19]) == 0 )
              {
                v66[19] = v64;
                goto LABEL_153;
              }
              do
              {
                v68 = v67;
                v69 = *(_QWORD *)(v67 + 112);
                if ( !v69 )
                  break;
                v68 = v69;
                v67 = *(_QWORD *)(v69 + 112);
              }
              while ( v67 );
            }
            *(_QWORD *)(v68 + 112) = v64;
LABEL_153:
            *(_QWORD *)(v64 + 112) = 0;
            if ( v6 )
              *(_QWORD *)(v6 + 56) = v64;
            v63 = *(_QWORD *)(v63 + 112);
            if ( !v63 )
              break;
            continue;
          }
        }
LABEL_163:
        result = *(_QWORD **)(a1 + 88);
        if ( result )
        {
          v70 = *(result - 3);
          result = v111;
          v71 = v111[11];
          if ( v71 )
          {
            *(_BYTE *)(v71 + 1) = (*(_BYTE *)(v71 + 1) | *(_BYTE *)(v70 + 1)) & 2 | *(_BYTE *)(v71 + 1) & 0xFD;
            v72 = *(_QWORD *)(v71 + 48);
            if ( v72 )
            {
              do
              {
                v73 = v72;
                v72 = *(_QWORD *)(v72 + 56);
              }
              while ( v72 );
              *(_QWORD *)(v73 + 56) = *(_QWORD *)(v70 + 48);
            }
            else
            {
              *(_QWORD *)(v71 + 48) = *(_QWORD *)(v70 + 48);
            }
            result = *(_QWORD **)(v70 + 24);
            if ( result )
            {
              do
              {
                v74 = result;
                result = (_QWORD *)result[4];
              }
              while ( result );
              v74[4] = *(_QWORD *)(v71 + 24);
              *(_QWORD *)(v71 + 24) = *(_QWORD *)(v70 + 24);
              for ( result = *(_QWORD **)(v70 + 48); result; result = (_QWORD *)result[7] )
              {
                if ( !result[5] )
                  result[5] = *(_QWORD *)(v71 + 24);
              }
            }
          }
          else
          {
            v111[11] = v70;
          }
        }
        return result;
      }
    }
    do
    {
      v47 = v46;
      v57 = (_QWORD **)*v46;
      if ( !v57 )
        break;
      v47 = v57;
      v46 = *v57;
    }
    while ( v46 );
LABEL_102:
    *v47 = v45;
    goto LABEL_96;
  }
  result = (_QWORD *)a1;
  v63 = *(_QWORD *)(a1 + 152);
  if ( v63 )
  {
    if ( v109 )
    {
      v64 = 0;
      if ( v6 )
        goto LABEL_150;
      goto LABEL_193;
    }
  }
  else
  {
    result = (_QWORD *)v109;
    if ( v109 )
      goto LABEL_163;
  }
  return result;
}

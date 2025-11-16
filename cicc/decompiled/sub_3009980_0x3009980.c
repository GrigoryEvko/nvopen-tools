// Function: sub_3009980
// Address: 0x3009980
//
__int64 __fastcall sub_3009980(__int64 a1, unsigned __int16 *a2)
{
  unsigned __int16 v4; // ax
  _OWORD *v6; // rax
  __int64 v7; // rdx
  __m128i v8; // xmm0
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __m128i si128; // xmm0
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // rdx
  __int64 v18; // rbx
  unsigned __int64 v19; // rcx
  unsigned __int8 v20; // r15
  __int8 *v21; // r14
  size_t v22; // r8
  _QWORD *v23; // rax
  unsigned __int64 v24; // rbx
  __int8 *v25; // r14
  unsigned __int64 v26; // rbx
  _QWORD *v27; // rax
  __m128i *v28; // rax
  unsigned __int64 *v29; // rax
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdi
  __m128i *v32; // rax
  __int64 v33; // rcx
  __m128i *v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  size_t v38; // rdx
  unsigned __int64 v39; // rdx
  char v40; // al
  unsigned __int64 v41; // rcx
  __int8 *v42; // r14
  unsigned __int64 v43; // rbx
  _QWORD *v44; // rax
  __m128i *v45; // rax
  __int64 v46; // rcx
  __m128i *v47; // rdx
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  char v50; // al
  unsigned __int64 v51; // rcx
  __int8 *v52; // r14
  unsigned __int64 v53; // rbx
  _QWORD *v54; // rax
  int v55; // edi
  unsigned __int64 v56; // rax
  char *v57; // rsi
  char *v58; // rax
  char *v59; // rdi
  unsigned __int64 v60; // rax
  _QWORD *v61; // rdi
  __int64 v62; // rax
  __int16 v63; // ax
  unsigned __int64 v64; // rdx
  int v65; // eax
  unsigned __int64 v66; // rcx
  __int8 *v67; // r14
  size_t v68; // r8
  __m128i *v69; // rax
  unsigned __int16 v70; // ax
  char *v71; // rcx
  size_t v72; // r8
  unsigned __int64 *v73; // rax
  unsigned __int64 v74; // rax
  unsigned __int64 v75; // rdi
  __m128i *v76; // rax
  __int64 v77; // rcx
  __m128i *v78; // rdx
  unsigned __int64 v79; // rax
  _QWORD *v80; // rdi
  unsigned __int64 v81; // rax
  char *v82; // rsi
  __int64 v83; // rax
  __m128i *v84; // rdi
  unsigned __int64 v85; // rdx
  bool v86; // al
  unsigned __int64 v87; // rax
  _QWORD *v88; // rdi
  size_t v89; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v90; // [rsp+18h] [rbp-B8h] BYREF
  _QWORD *v91; // [rsp+20h] [rbp-B0h] BYREF
  unsigned __int64 v92; // [rsp+28h] [rbp-A8h]
  _QWORD v93[2]; // [rsp+30h] [rbp-A0h] BYREF
  __m128i *v94; // [rsp+40h] [rbp-90h] BYREF
  size_t v95; // [rsp+48h] [rbp-88h]
  __m128i v96; // [rsp+50h] [rbp-80h] BYREF
  __m128i *v97; // [rsp+60h] [rbp-70h] BYREF
  size_t v98; // [rsp+68h] [rbp-68h]
  __m128i v99; // [rsp+70h] [rbp-60h] BYREF
  __int64 v100; // [rsp+80h] [rbp-50h] BYREF
  size_t v101; // [rsp+88h] [rbp-48h]
  _QWORD v102[8]; // [rsp+90h] [rbp-40h] BYREF

  v4 = *a2;
  if ( *a2 > 0x111u )
  {
    if ( v4 == 505 )
    {
      *(_QWORD *)(a1 + 8) = 8;
      *(_QWORD *)a1 = a1 + 16;
      strcpy((char *)(a1 + 16), "Metadata");
      return a1;
    }
    v16 = v4;
  }
  else
  {
    if ( v4 > 0x104u )
    {
      switch ( v4 )
      {
        case 0x106u:
          strcpy((char *)(a1 + 16), "glue");
          *(_QWORD *)a1 = a1 + 16;
          *(_QWORD *)(a1 + 8) = 4;
          break;
        case 0x107u:
          *(_QWORD *)a1 = a1 + 16;
          strcpy((char *)(a1 + 16), "isVoid");
          *(_QWORD *)(a1 + 8) = 6;
          break;
        case 0x108u:
          *(_BYTE *)(a1 + 22) = 100;
          *(_QWORD *)a1 = a1 + 16;
          *(_DWORD *)(a1 + 16) = 2037673557;
          *(_WORD *)(a1 + 20) = 25968;
          *(_QWORD *)(a1 + 8) = 7;
          *(_BYTE *)(a1 + 23) = 0;
          break;
        case 0x109u:
          *(_BYTE *)(a1 + 22) = 102;
          *(_QWORD *)a1 = a1 + 16;
          *(_DWORD *)(a1 + 16) = 1668183398;
          *(_WORD *)(a1 + 20) = 25970;
          *(_QWORD *)(a1 + 8) = 7;
          *(_BYTE *)(a1 + 23) = 0;
          break;
        case 0x10Au:
          *(_QWORD *)a1 = a1 + 16;
          strcpy((char *)(a1 + 16), "externref");
          *(_QWORD *)(a1 + 8) = 9;
          break;
        case 0x10Bu:
          strcpy((char *)(a1 + 16), "exnref");
          *(_QWORD *)a1 = a1 + 16;
          *(_QWORD *)(a1 + 8) = 6;
          break;
        case 0x10Cu:
          *(_QWORD *)a1 = a1 + 16;
          strcpy((char *)(a1 + 16), "x86amx");
          *(_QWORD *)(a1 + 8) = 6;
          break;
        case 0x10Du:
          *(_DWORD *)(a1 + 16) = 2016687721;
          *(_QWORD *)a1 = a1 + 16;
          *(_BYTE *)(a1 + 20) = 56;
          *(_QWORD *)(a1 + 8) = 5;
          *(_BYTE *)(a1 + 21) = 0;
          break;
        case 0x10Eu:
          *(_QWORD *)a1 = a1 + 16;
          strcpy((char *)(a1 + 16), "aarch64svcount");
          *(_QWORD *)(a1 + 8) = 14;
          break;
        case 0x10Fu:
          *(_QWORD *)a1 = a1 + 16;
          strcpy((char *)(a1 + 16), "spirvbuiltin");
          *(_QWORD *)(a1 + 8) = 12;
          break;
        case 0x110u:
          *(_QWORD *)a1 = a1 + 16;
          v100 = 22;
          v11 = sub_22409D0(a1, (unsigned __int64 *)&v100, 0);
          v12 = v100;
          si128 = _mm_load_si128((const __m128i *)&xmmword_4457AD0);
          *(_QWORD *)a1 = v11;
          *(_QWORD *)(a1 + 16) = v12;
          *(_DWORD *)(v11 + 16) = 1953393007;
          *(_WORD *)(v11 + 20) = 29285;
          *(__m128i *)v11 = si128;
          v14 = v100;
          v15 = *(_QWORD *)a1;
          *(_QWORD *)(a1 + 8) = v100;
          *(_BYTE *)(v15 + v14) = 0;
          break;
        case 0x111u:
          *(_QWORD *)a1 = a1 + 16;
          v100 = 26;
          v6 = (_OWORD *)sub_22409D0(a1, (unsigned __int64 *)&v100, 0);
          v7 = v100;
          v8 = _mm_load_si128((const __m128i *)&xmmword_4457AE0);
          *(_QWORD *)a1 = v6;
          *(_QWORD *)(a1 + 16) = v7;
          qmemcpy(v6 + 1, "dedPointer", 10);
          *v6 = v8;
          v9 = v100;
          v10 = *(_QWORD *)a1;
          *(_QWORD *)(a1 + 8) = v100;
          *(_BYTE *)(v10 + v9) = 0;
          break;
        default:
          *(_QWORD *)a1 = a1 + 16;
          strcpy((char *)(a1 + 16), "x86mmx");
          *(_QWORD *)(a1 + 8) = 6;
          break;
      }
      return a1;
    }
    switch ( v4 )
    {
      case 0xAu:
        strcpy((char *)(a1 + 16), "bf16");
        *(_QWORD *)a1 = a1 + 16;
        *(_QWORD *)(a1 + 8) = 4;
        return a1;
      case 0x10u:
        *(_BYTE *)(a1 + 22) = 56;
        *(_QWORD *)a1 = a1 + 16;
        *(_DWORD *)(a1 + 16) = 1717792880;
        *(_WORD *)(a1 + 20) = 12849;
        *(_QWORD *)(a1 + 8) = 7;
        *(_BYTE *)(a1 + 23) = 0;
        return a1;
      case 1u:
        *(_QWORD *)(a1 + 8) = 2;
        *(_QWORD *)a1 = a1 + 16;
        strcpy((char *)(a1 + 16), "ch");
        return a1;
    }
    v16 = v4;
    if ( (unsigned __int16)(v4 - 229) <= 0x1Fu )
    {
      v17 = v4 - 1;
      v18 = *(_QWORD *)&byte_444C4A0[16 * v17];
      v19 = byte_4457900[v17];
      v20 = byte_4457900[v17];
      if ( !v20 )
      {
        v99.m128i_i8[4] = 48;
        v21 = &v99.m128i_i8[4];
        v100 = (__int64)v102;
LABEL_29:
        v22 = 1;
        LOBYTE(v102[0]) = *v21;
        v23 = v102;
        goto LABEL_30;
      }
      v21 = &v99.m128i_i8[5];
      do
      {
        --v21;
        v55 = v19 % 0xA;
        v56 = v19;
        v19 /= 0xAu;
        *v21 = v55 + 48;
      }
      while ( v56 > 9 );
      v57 = (char *)(&v99.m128i_u8[5] - (unsigned __int8 *)v21);
      v100 = (__int64)v102;
      v22 = &v99.m128i_u8[5] - (unsigned __int8 *)v21;
      v94 = (__m128i *)(&v99.m128i_u8[5] - (unsigned __int8 *)v21);
      if ( (unsigned __int64)(&v99.m128i_u8[5] - (unsigned __int8 *)v21) <= 0xF )
      {
        if ( v57 == (char *)1 )
          goto LABEL_29;
        if ( !v57 )
        {
          v23 = v102;
LABEL_30:
          v101 = v22;
          *((_BYTE *)v23 + v22) = 0;
          v24 = (unsigned int)v18 / (8 * (unsigned int)v20);
          if ( !v24 )
          {
            v99.m128i_i8[4] = 48;
            v25 = &v99.m128i_i8[4];
            v91 = v93;
LABEL_32:
            v26 = 1;
            LOBYTE(v93[0]) = *v25;
            v27 = v93;
            goto LABEL_33;
          }
          v25 = &v99.m128i_i8[5];
          do
          {
            *--v25 = v24 % 0xA + 48;
            v79 = v24;
            v24 /= 0xAu;
          }
          while ( v79 > 9 );
          v26 = &v99.m128i_u8[5] - (unsigned __int8 *)v25;
          v91 = v93;
          v94 = (__m128i *)(&v99.m128i_u8[5] - (unsigned __int8 *)v25);
          if ( (unsigned __int64)(&v99.m128i_u8[5] - (unsigned __int8 *)v25) <= 0xF )
          {
            if ( v26 == 1 )
              goto LABEL_32;
            if ( !v26 )
            {
              v27 = v93;
LABEL_33:
              v92 = v26;
              *((_BYTE *)v27 + v26) = 0;
              v28 = (__m128i *)sub_2241130((unsigned __int64 *)&v91, 0, 0, "riscv_nxv", 9u);
              v94 = &v96;
              if ( (__m128i *)v28->m128i_i64[0] == &v28[1] )
              {
                v96 = _mm_loadu_si128(v28 + 1);
              }
              else
              {
                v94 = (__m128i *)v28->m128i_i64[0];
                v96.m128i_i64[0] = v28[1].m128i_i64[0];
              }
              v95 = v28->m128i_u64[1];
              v28->m128i_i64[0] = (__int64)v28[1].m128i_i64;
              v28->m128i_i64[1] = 0;
              v28[1].m128i_i8[0] = 0;
              if ( 0x3FFFFFFFFFFFFFFFLL - v95 <= 2 )
                sub_4262D8((__int64)"basic_string::append");
              v29 = sub_2241490((unsigned __int64 *)&v94, "i8x", 3u);
              v97 = &v99;
              if ( (unsigned __int64 *)*v29 == v29 + 2 )
              {
                v99 = _mm_loadu_si128((const __m128i *)v29 + 1);
              }
              else
              {
                v97 = (__m128i *)*v29;
                v99.m128i_i64[0] = v29[2];
              }
              v98 = v29[1];
              *v29 = (unsigned __int64)(v29 + 2);
              v29[1] = 0;
              *((_BYTE *)v29 + 16) = 0;
              v30 = 15;
              v31 = 15;
              if ( v97 != &v99 )
                v31 = v99.m128i_i64[0];
              if ( v98 + v101 <= v31 )
                goto LABEL_44;
              if ( (_QWORD *)v100 != v102 )
                v30 = v102[0];
              if ( v98 + v101 <= v30 )
              {
                v32 = (__m128i *)sub_2241130((unsigned __int64 *)&v100, 0, 0, v97, v98);
                *(_QWORD *)a1 = a1 + 16;
                v33 = v32->m128i_i64[0];
                v34 = v32 + 1;
                if ( (__m128i *)v32->m128i_i64[0] != &v32[1] )
                  goto LABEL_45;
              }
              else
              {
LABEL_44:
                v32 = (__m128i *)sub_2241490((unsigned __int64 *)&v97, (char *)v100, v101);
                *(_QWORD *)a1 = a1 + 16;
                v33 = v32->m128i_i64[0];
                v34 = v32 + 1;
                if ( (__m128i *)v32->m128i_i64[0] != &v32[1] )
                {
LABEL_45:
                  *(_QWORD *)a1 = v33;
                  *(_QWORD *)(a1 + 16) = v32[1].m128i_i64[0];
LABEL_46:
                  *(_QWORD *)(a1 + 8) = v32->m128i_i64[1];
                  v32->m128i_i64[0] = (__int64)v34;
                  v32->m128i_i64[1] = 0;
                  v32[1].m128i_i8[0] = 0;
                  if ( v97 != &v99 )
                    j_j___libc_free_0((unsigned __int64)v97);
                  if ( v94 != &v96 )
                    j_j___libc_free_0((unsigned __int64)v94);
                  if ( v91 != v93 )
                    j_j___libc_free_0((unsigned __int64)v91);
                  goto LABEL_52;
                }
              }
              *(__m128i *)(a1 + 16) = _mm_loadu_si128(v32 + 1);
              goto LABEL_46;
            }
            v80 = v93;
          }
          else
          {
            v91 = (_QWORD *)sub_22409D0((__int64)&v91, (unsigned __int64 *)&v94, 0);
            v80 = v91;
            v93[0] = v94;
          }
          memcpy(v80, v25, &v99.m128i_u8[5] - (unsigned __int8 *)v25);
          v26 = (unsigned __int64)v94;
          v27 = v91;
          goto LABEL_33;
        }
        v59 = (char *)v102;
      }
      else
      {
        v58 = (char *)sub_22409D0((__int64)&v100, (unsigned __int64 *)&v94, 0);
        v22 = &v99.m128i_u8[5] - (unsigned __int8 *)v21;
        v100 = (__int64)v58;
        v59 = v58;
        v102[0] = v94;
      }
      memcpy(v59, v21, v22);
      v22 = (size_t)v94;
      v23 = (_QWORD *)v100;
      goto LABEL_30;
    }
    if ( !v4 )
    {
      if ( !sub_30070B0((__int64)a2) )
      {
        if ( !sub_3007070((__int64)a2) )
        {
          if ( (unsigned __int8)sub_3007030((__int64)a2) )
          {
            v94 = (__m128i *)sub_3007260((__int64)a2);
            v95 = v38;
            v39 = (unsigned __int64)v94;
            v40 = v95;
            goto LABEL_59;
          }
LABEL_163:
          BUG();
        }
        v91 = (_QWORD *)sub_3007260((__int64)a2);
        v92 = v85;
        v49 = (unsigned __int64)v91;
        v50 = v92;
LABEL_70:
        v94 = (__m128i *)v49;
        LOBYTE(v95) = v50;
        v51 = sub_CA1930(&v94);
        if ( !v51 )
        {
          v99.m128i_i8[4] = 48;
          v52 = &v99.m128i_i8[4];
          v100 = (__int64)v102;
LABEL_72:
          v53 = 1;
          LOBYTE(v102[0]) = *v52;
          v54 = v102;
          goto LABEL_73;
        }
        v52 = &v99.m128i_i8[5];
        do
        {
          *--v52 = v51 % 0xA + 48;
          v87 = v51;
          v51 /= 0xAu;
        }
        while ( v87 > 9 );
        v53 = &v99.m128i_u8[5] - (unsigned __int8 *)v52;
        v100 = (__int64)v102;
        v90 = &v99.m128i_u8[5] - (unsigned __int8 *)v52;
        if ( (unsigned __int64)(&v99.m128i_u8[5] - (unsigned __int8 *)v52) <= 0xF )
        {
          if ( v53 == 1 )
            goto LABEL_72;
          if ( !v53 )
          {
            v54 = v102;
LABEL_73:
            v101 = v53;
            *((_BYTE *)v54 + v53) = 0;
            v45 = (__m128i *)sub_2241130((unsigned __int64 *)&v100, 0, 0, "i", 1u);
            *(_QWORD *)a1 = a1 + 16;
            v46 = v45->m128i_i64[0];
            v47 = v45 + 1;
            if ( (__m128i *)v45->m128i_i64[0] != &v45[1] )
              goto LABEL_63;
LABEL_74:
            *(__m128i *)(a1 + 16) = _mm_loadu_si128(v45 + 1);
            goto LABEL_64;
          }
          v88 = v102;
        }
        else
        {
          v100 = sub_22409D0((__int64)&v100, (unsigned __int64 *)&v90, 0);
          v88 = (_QWORD *)v100;
          v102[0] = v90;
        }
        memcpy(v88, v52, &v99.m128i_u8[5] - (unsigned __int8 *)v52);
        v53 = v90;
        v54 = (_QWORD *)v100;
        goto LABEL_73;
      }
      v63 = sub_3009970((__int64)a2, (__int64)a2, v35, v36, v37);
LABEL_89:
      LOWORD(v91) = v63;
      v92 = v64;
      sub_3009980(&v100, &v91);
      v65 = *a2;
      if ( (_WORD)v65 )
      {
        LODWORD(v66) = word_4456340[v65 - 1];
      }
      else
      {
        v90 = sub_3007240((__int64)a2);
        LODWORD(v66) = v90;
      }
      v66 = (unsigned int)v66;
      if ( !(_DWORD)v66 )
      {
        v99.m128i_i8[4] = 48;
        v67 = &v99.m128i_i8[4];
        v94 = &v96;
LABEL_93:
        v68 = 1;
        v96.m128i_i8[0] = *v67;
        v69 = &v96;
        goto LABEL_94;
      }
      v67 = &v99.m128i_i8[5];
      do
      {
        *--v67 = v66 % 0xA + 48;
        v81 = v66;
        v66 /= 0xAu;
      }
      while ( v81 > 9 );
      v82 = (char *)(&v99.m128i_u8[5] - (unsigned __int8 *)v67);
      v94 = &v96;
      v68 = &v99.m128i_u8[5] - (unsigned __int8 *)v67;
      v89 = &v99.m128i_u8[5] - (unsigned __int8 *)v67;
      if ( (unsigned __int64)(&v99.m128i_u8[5] - (unsigned __int8 *)v67) <= 0xF )
      {
        if ( v82 == (char *)1 )
          goto LABEL_93;
        if ( !v82 )
        {
          v69 = &v96;
LABEL_94:
          v95 = v68;
          v69->m128i_i8[v68] = 0;
          if ( *a2 )
          {
            v70 = *a2 - 176;
            v71 = "v";
            v72 = v70 < 0x35u ? 3LL : 1LL;
            if ( v70 <= 0x34u )
              v71 = "nxv";
          }
          else
          {
            v86 = sub_3007100((__int64)a2);
            v71 = "v";
            v72 = (-(__int64)!v86 & 0xFFFFFFFFFFFFFFFELL) + 3;
            if ( v86 )
              v71 = "nxv";
          }
          v73 = sub_2241130((unsigned __int64 *)&v94, 0, 0, v71, v72);
          v97 = &v99;
          if ( (unsigned __int64 *)*v73 == v73 + 2 )
          {
            v99 = _mm_loadu_si128((const __m128i *)v73 + 1);
          }
          else
          {
            v97 = (__m128i *)*v73;
            v99.m128i_i64[0] = v73[2];
          }
          v98 = v73[1];
          *v73 = (unsigned __int64)(v73 + 2);
          v73[1] = 0;
          *((_BYTE *)v73 + 16) = 0;
          v74 = 15;
          v75 = 15;
          if ( v97 != &v99 )
            v75 = v99.m128i_i64[0];
          if ( v98 + v101 <= v75 )
            goto LABEL_105;
          if ( (_QWORD *)v100 != v102 )
            v74 = v102[0];
          if ( v98 + v101 <= v74 )
          {
            v76 = (__m128i *)sub_2241130((unsigned __int64 *)&v100, 0, 0, v97, v98);
            *(_QWORD *)a1 = a1 + 16;
            v77 = v76->m128i_i64[0];
            v78 = v76 + 1;
            if ( (__m128i *)v76->m128i_i64[0] != &v76[1] )
              goto LABEL_106;
          }
          else
          {
LABEL_105:
            v76 = (__m128i *)sub_2241490((unsigned __int64 *)&v97, (char *)v100, v101);
            *(_QWORD *)a1 = a1 + 16;
            v77 = v76->m128i_i64[0];
            v78 = v76 + 1;
            if ( (__m128i *)v76->m128i_i64[0] != &v76[1] )
            {
LABEL_106:
              *(_QWORD *)a1 = v77;
              *(_QWORD *)(a1 + 16) = v76[1].m128i_i64[0];
LABEL_107:
              *(_QWORD *)(a1 + 8) = v76->m128i_i64[1];
              v76->m128i_i64[0] = (__int64)v78;
              v76->m128i_i64[1] = 0;
              v76[1].m128i_i8[0] = 0;
              if ( v97 != &v99 )
                j_j___libc_free_0((unsigned __int64)v97);
              if ( v94 != &v96 )
                j_j___libc_free_0((unsigned __int64)v94);
              goto LABEL_52;
            }
          }
          *(__m128i *)(a1 + 16) = _mm_loadu_si128(v76 + 1);
          goto LABEL_107;
        }
        v84 = &v96;
      }
      else
      {
        v83 = sub_22409D0((__int64)&v94, &v89, 0);
        v68 = &v99.m128i_u8[5] - (unsigned __int8 *)v67;
        v94 = (__m128i *)v83;
        v84 = (__m128i *)v83;
        v96.m128i_i64[0] = v89;
      }
      memcpy(v84, v67, v68);
      v68 = v89;
      v69 = v94;
      goto LABEL_94;
    }
  }
  if ( (unsigned __int16)(v4 - 17) <= 0xD3u )
  {
    v63 = word_4456580[v16 - 1];
    v64 = 0;
    goto LABEL_89;
  }
  if ( (unsigned __int16)(v4 - 2) <= 7u )
  {
    if ( (unsigned __int16)(v4 - 504) <= 7u )
      goto LABEL_163;
    v48 = 16LL * (v16 - 1);
    v49 = *(_QWORD *)&byte_444C4A0[v48];
    v50 = byte_444C4A0[v48 + 8];
    goto LABEL_70;
  }
  if ( (unsigned __int16)(v4 - 10) > 6u || (unsigned __int16)(v4 - 504) <= 7u )
    goto LABEL_163;
  v62 = 16LL * (v16 - 1);
  v39 = *(_QWORD *)&byte_444C4A0[v62];
  v40 = byte_444C4A0[v62 + 8];
LABEL_59:
  v91 = (_QWORD *)v39;
  LOBYTE(v92) = v40;
  v41 = sub_CA1930(&v91);
  if ( !v41 )
  {
    v99.m128i_i8[4] = 48;
    v42 = &v99.m128i_i8[4];
    v100 = (__int64)v102;
LABEL_61:
    v43 = 1;
    LOBYTE(v102[0]) = *v42;
    v44 = v102;
    goto LABEL_62;
  }
  v42 = &v99.m128i_i8[5];
  do
  {
    *--v42 = v41 % 0xA + 48;
    v60 = v41;
    v41 /= 0xAu;
  }
  while ( v60 > 9 );
  v43 = &v99.m128i_u8[5] - (unsigned __int8 *)v42;
  v100 = (__int64)v102;
  v90 = &v99.m128i_u8[5] - (unsigned __int8 *)v42;
  if ( (unsigned __int64)(&v99.m128i_u8[5] - (unsigned __int8 *)v42) > 0xF )
  {
    v100 = sub_22409D0((__int64)&v100, (unsigned __int64 *)&v90, 0);
    v61 = (_QWORD *)v100;
    v102[0] = v90;
LABEL_84:
    memcpy(v61, v42, &v99.m128i_u8[5] - (unsigned __int8 *)v42);
    v43 = v90;
    v44 = (_QWORD *)v100;
    goto LABEL_62;
  }
  if ( v43 == 1 )
    goto LABEL_61;
  if ( v43 )
  {
    v61 = v102;
    goto LABEL_84;
  }
  v44 = v102;
LABEL_62:
  v101 = v43;
  *((_BYTE *)v44 + v43) = 0;
  v45 = (__m128i *)sub_2241130((unsigned __int64 *)&v100, 0, 0, "f", 1u);
  *(_QWORD *)a1 = a1 + 16;
  v46 = v45->m128i_i64[0];
  v47 = v45 + 1;
  if ( (__m128i *)v45->m128i_i64[0] == &v45[1] )
    goto LABEL_74;
LABEL_63:
  *(_QWORD *)a1 = v46;
  *(_QWORD *)(a1 + 16) = v45[1].m128i_i64[0];
LABEL_64:
  *(_QWORD *)(a1 + 8) = v45->m128i_i64[1];
  v45->m128i_i64[0] = (__int64)v47;
  v45->m128i_i64[1] = 0;
  v45[1].m128i_i8[0] = 0;
LABEL_52:
  if ( (_QWORD *)v100 != v102 )
    j_j___libc_free_0(v100);
  return a1;
}

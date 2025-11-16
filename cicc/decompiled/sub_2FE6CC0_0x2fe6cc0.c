// Function: sub_2FE6CC0
// Address: 0x2fe6cc0
//
__int64 __fastcall sub_2FE6CC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  int v8; // edx
  char v9; // cl
  __int16 v10; // ax
  __int64 v12; // r12
  __int16 v13; // ax
  __int64 v14; // rdx
  __int16 v15; // bx
  __int64 v16; // rax
  unsigned __int32 v17; // r14d
  unsigned __int64 v18; // rsi
  __int64 v19; // r12
  unsigned __int64 v20; // rsi
  __int64 v21; // rsi
  __int16 v22; // ax
  __int64 v23; // rdx
  __int64 v24; // rdx
  unsigned int v25; // eax
  __m128i v26; // rax
  unsigned __int64 v27; // rax
  __int16 v28; // ax
  __int64 v29; // rdx
  __int16 v30; // bx
  char v31; // di
  __int64 v32; // rax
  unsigned __int64 v33; // rbx
  unsigned int v34; // esi
  __m128i v35; // xmm2
  unsigned __int16 v36; // ax
  unsigned __int64 v37; // rsi
  unsigned __int16 v38; // ax
  __int64 v39; // rdx
  unsigned int v40; // ebx
  __int32 v41; // esi
  __int16 v42; // ax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r15
  unsigned int v47; // eax
  unsigned __int16 v48; // cx
  __int64 v49; // rbx
  __int64 v50; // rax
  int v51; // eax
  int v52; // eax
  __int64 v53; // rdx
  unsigned __int64 v54; // rax
  unsigned int v55; // eax
  int v56; // r12d
  unsigned __int16 v57; // ax
  __int64 v58; // rdx
  __int64 v59; // r15
  unsigned __int32 v60; // r14d
  int v61; // esi
  __int64 v62; // r12
  unsigned int v63; // edx
  unsigned int v64; // r14d
  __m128i v65; // xmm1
  int v66; // eax
  int v67; // ecx
  __int64 v68; // rdx
  unsigned __int16 i; // ax
  unsigned __int16 v70; // ax
  __int64 v71; // rdx
  __int64 v72; // rdx
  char v73; // al
  int v74; // eax
  int v75; // eax
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // rdx
  __int64 v79; // rax
  __int64 v80; // rdx
  unsigned int v81; // eax
  __int16 v82; // di
  __int64 v83; // rax
  int v84; // eax
  int v85; // ecx
  __int64 v86; // rdx
  unsigned __int32 v87; // r15d
  __int64 v88; // r14
  __int16 v89; // ax
  __int64 v90; // rdx
  __m128i v91; // xmm3
  unsigned __int16 v92; // r14
  __int64 v93; // rdx
  __int64 v94; // r12
  __int64 v95; // rax
  unsigned int v96; // esi
  char v97; // dl
  int v98; // esi
  __int64 v99; // rax
  __int64 v100; // [rsp+0h] [rbp-130h]
  char v101; // [rsp+8h] [rbp-128h]
  __int8 v103; // [rsp+1Fh] [rbp-111h]
  __int64 v104; // [rsp+20h] [rbp-110h] BYREF
  __int64 v105; // [rsp+28h] [rbp-108h]
  __int64 v106; // [rsp+38h] [rbp-F8h]
  __int64 v107; // [rsp+40h] [rbp-F0h]
  __m128i v108; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v109; // [rsp+80h] [rbp-B0h] BYREF
  char v110; // [rsp+88h] [rbp-A8h]
  unsigned __int64 v111; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v112; // [rsp+98h] [rbp-98h]
  __int64 v113; // [rsp+A0h] [rbp-90h]
  __int64 v114; // [rsp+A8h] [rbp-88h]
  __int64 v115; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v116; // [rsp+B8h] [rbp-78h]
  __int64 v117; // [rsp+C0h] [rbp-70h]
  __int64 v118; // [rsp+C8h] [rbp-68h]
  __int64 v119; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v120; // [rsp+D8h] [rbp-58h]
  __m128i v121; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v122; // [rsp+F0h] [rbp-40h]

  v5 = (unsigned __int16)a4;
  v104 = a4;
  v105 = a5;
  if ( (_WORD)a4 )
  {
    v8 = (unsigned __int16)a4;
    v9 = *(_BYTE *)(a2 + (unsigned __int16)a4 + 524896);
    if ( v9 == 6 )
    {
      v38 = v5 - 176;
      v39 = v8 - 1;
      v40 = (unsigned __int16)word_4456580[v39];
      v41 = word_4456340[v39] >> 1;
      v121.m128i_i32[0] = v41;
      v121.m128i_i8[4] = v38 <= 0x34u;
      if ( v38 > 0x34u )
        v42 = sub_2D43050(v40, v41);
      else
        v42 = sub_2D43AD0(v40, v41);
      v43 = 0;
      if ( !v42 )
        v42 = sub_3009450(a3, v40, 0, v121.m128i_i64[0]);
      goto LABEL_52;
    }
    if ( v9 == 5 )
    {
      *(_BYTE *)a1 = 5;
      *(_QWORD *)(a1 + 16) = 0;
      v10 = word_4456580[(unsigned __int16)v5 - 1];
    }
    else
    {
      v10 = *(_WORD *)(a2 + 2 * v5 + 5866);
      *(_BYTE *)a1 = v9;
      *(_QWORD *)(a1 + 16) = 0;
    }
    *(_WORD *)(a1 + 8) = v10;
    return a1;
  }
  if ( !(unsigned __int8)sub_30070B0(&v104, a2, a3) )
  {
    v113 = sub_3007260(&v104);
    v114 = v24;
    v121.m128i_i64[0] = v113;
    v121.m128i_i8[8] = v24;
    v25 = sub_CA1930(&v121);
    if ( v25 > 7 && (v25 & (v25 - 1)) == 0 )
    {
      if ( !(_WORD)v104 )
      {
        v26.m128i_i64[0] = sub_3007260(&v104);
        v121 = v26;
LABEL_25:
        v119 = v26.m128i_i64[0];
        LOBYTE(v120) = v26.m128i_i8[8];
        v27 = (unsigned __int64)sub_CA1930(&v119) >> 1;
        switch ( (_DWORD)v27 )
        {
          case 1:
            v28 = 2;
            v29 = 0;
            break;
          case 2:
            v28 = 3;
            v29 = 0;
            break;
          case 4:
            v28 = 4;
            v29 = 0;
            break;
          case 8:
            v28 = 5;
            v29 = 0;
            break;
          case 0x10:
            v28 = 6;
            v29 = 0;
            break;
          case 0x20:
            v28 = 7;
            v29 = 0;
            break;
          case 0x40:
            v28 = 8;
            v29 = 0;
            break;
          case 0x80:
            v28 = 9;
            v29 = 0;
            break;
          default:
            v28 = sub_3007020(a3, (unsigned int)v27);
            break;
        }
        *(_BYTE *)a1 = 2;
        *(_WORD *)(a1 + 8) = v28;
        *(_QWORD *)(a1 + 16) = v29;
        return a1;
      }
      if ( (_WORD)v104 != 1 && (unsigned __int16)(v104 - 504) > 7u )
      {
        v26.m128i_i64[1] = 16LL * ((unsigned __int16)v104 - 1);
        v26.m128i_i64[0] = *(_QWORD *)&byte_444C4A0[v26.m128i_i64[1]];
        v26.m128i_i8[8] = byte_444C4A0[v26.m128i_i64[1] + 8];
        goto LABEL_25;
      }
LABEL_178:
      BUG();
    }
    if ( (_WORD)v104 )
    {
      if ( (_WORD)v104 == 1 || (unsigned __int16)(v104 - 504) <= 7u )
        goto LABEL_178;
      v45 = 16LL * ((unsigned __int16)v104 - 1);
      v44 = *(_QWORD *)&byte_444C4A0[v45];
      LOBYTE(v45) = byte_444C4A0[v45 + 8];
    }
    else
    {
      v44 = sub_3007260(&v104);
      v115 = v44;
      v116 = v45;
    }
    v121.m128i_i64[0] = v44;
    v46 = 0;
    v121.m128i_i8[8] = v45;
    v47 = sub_CA1930(&v121);
    v48 = 5;
    if ( v47 > 8 )
    {
      _BitScanReverse(&v47, v47 - 1);
      v66 = v47 ^ 0x1F;
      v67 = 32 - v66;
      if ( v66 == 28 )
      {
        v48 = 6;
      }
      else
      {
        switch ( v67 )
        {
          case 5:
            v48 = 7;
            break;
          case 6:
            v48 = 8;
            break;
          case 7:
            v48 = 9;
            break;
          default:
            v48 = sub_3007020(a3, (unsigned int)(1 << (32 - v66)));
            v46 = v68;
            goto LABEL_56;
        }
      }
      v46 = 0;
    }
LABEL_56:
    v49 = v48;
    sub_2FE6CC0(&v121, a2, a3, v48, v46);
    if ( v121.m128i_i8[0] == 1 )
    {
      v50 = v122;
      *(__m128i *)a1 = _mm_loadu_si128(&v121);
      *(_QWORD *)(a1 + 16) = v50;
    }
    else
    {
      *(_BYTE *)a1 = 1;
      *(_QWORD *)(a1 + 8) = v49;
      *(_QWORD *)(a1 + 16) = v46;
    }
    return a1;
  }
  v12 = sub_3007240(&v104);
  v106 = v12;
  v103 = BYTE4(v12);
  v13 = sub_3009970(&v104);
  v108.m128i_i64[1] = v14;
  v15 = v13;
  v108.m128i_i16[0] = v13;
  v101 = ((_DWORD)v12 == 1) & (BYTE4(v12) ^ 1);
  if ( v101 )
  {
    v65 = _mm_loadu_si128(&v108);
    *(_BYTE *)a1 = 5;
    *(__m128i *)(a1 + 8) = v65;
    return a1;
  }
  if ( v13 )
  {
    if ( (unsigned __int16)(v13 - 2) > 7u
      && (unsigned __int16)(v13 - 17) > 0x6Cu
      && (unsigned __int16)(v13 - 176) > 0x1Fu )
    {
      goto LABEL_45;
    }
  }
  else if ( !(unsigned __int8)sub_3007070(&v108) )
  {
    goto LABEL_36;
  }
  if ( (_WORD)v104 )
  {
    LODWORD(v16) = word_4456340[(unsigned __int16)v104 - 1];
  }
  else
  {
    v16 = sub_3007240(&v104);
    v107 = v16;
  }
  if ( ((unsigned int)v16 & ((_DWORD)v16 - 1)) != 0 )
  {
    v17 = v108.m128i_i32[0];
    v18 = (((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
        | (unsigned int)v12
        | ((unsigned __int64)(unsigned int)v12 >> 1);
    v19 = v108.m128i_i64[1];
    v20 = (((v18 >> 4) | v18) >> 8) | (v18 >> 4) | v18;
    v21 = ((v20 >> 16) | v20) + 1;
    v121.m128i_i32[0] = v21;
    v121.m128i_i8[4] = v103;
    if ( v103 )
      v22 = sub_2D43AD0(v108.m128i_i16[0], v21);
    else
      v22 = sub_2D43050(v108.m128i_i16[0], v21);
    v23 = 0;
    if ( !v22 )
      v22 = sub_3009450(a3, v17, v19, v121.m128i_i64[0]);
    *(_BYTE *)a1 = 7;
    *(_WORD *)(a1 + 8) = v22;
    *(_QWORD *)(a1 + 16) = v23;
    return a1;
  }
  sub_2FE6CC0(&v121, a2, a3, v108.m128i_u32[0], v108.m128i_i64[1]);
  if ( v121.m128i_i8[0] == 2 )
  {
    if ( (_WORD)v104 )
    {
      if ( (unsigned __int16)(v104 - 176) <= 0x34u )
      {
LABEL_152:
        v91 = _mm_loadu_si128(&v108);
        *(_BYTE *)a1 = 10;
        *(__m128i *)(a1 + 8) = v91;
        return a1;
      }
      v94 = 0;
      v99 = (unsigned __int16)v104 - 1;
      v92 = word_4456580[v99];
      v97 = 0;
    }
    else
    {
      if ( (unsigned __int8)((unsigned __int64)sub_3007240(&v104) >> 32) )
        goto LABEL_152;
      v92 = sub_3009970(&v104);
      v94 = v93;
      if ( !(_WORD)v104 )
      {
        v95 = sub_3007240(&v104);
        v96 = v95;
        v101 = BYTE4(v95);
        v97 = BYTE4(v95);
        goto LABEL_160;
      }
      v101 = (unsigned __int16)(v104 - 176) <= 0x34u;
      v97 = v101;
      v99 = (unsigned __int16)v104 - 1;
    }
    v96 = word_4456340[v99];
LABEL_160:
    v98 = v96 >> 1;
    BYTE4(v119) = v97;
    LODWORD(v119) = v98;
    if ( v101 )
      v42 = sub_2D43AD0(v92, v98);
    else
      v42 = sub_2D43050(v92, v98);
    v43 = 0;
    if ( !v42 )
      v42 = sub_3009450(a3, v92, v94, v119);
    goto LABEL_52;
  }
  v15 = v108.m128i_i16[0];
  v100 = v108.m128i_i64[1];
  for ( i = v108.m128i_i16[0]; ; i = v108.m128i_i16[0] )
  {
    if ( i )
    {
      if ( i == 1 || (unsigned __int16)(i - 504) <= 7u )
        goto LABEL_178;
      v83 = 16LL * (i - 1);
      v72 = *(_QWORD *)&byte_444C4A0[v83];
      v73 = byte_444C4A0[v83 + 8];
    }
    else
    {
      v117 = sub_3007260(&v108);
      v118 = v71;
      v72 = v117;
      v73 = v118;
    }
    v109 = v72;
    v110 = v73;
    v74 = sub_CA1930(&v109);
    if ( v74 )
    {
      switch ( v74 )
      {
        case 1:
          v75 = 3;
          v112 = 0;
          LOWORD(v111) = 3;
          break;
        case 3:
          LOWORD(v111) = 4;
          v75 = 4;
          v112 = 0;
          break;
        case 7:
          v75 = 5;
          v112 = 0;
          LOWORD(v111) = 5;
          break;
        case 15:
          v75 = 6;
          v112 = 0;
          LOWORD(v111) = 6;
          break;
        case 31:
          v75 = 7;
          v112 = 0;
          LOWORD(v111) = 7;
          break;
        case 63:
          v75 = 8;
          v112 = 0;
          LOWORD(v111) = 8;
          break;
        case 127:
          v112 = 0;
          LOWORD(v111) = 9;
          v75 = 9;
          break;
        default:
          LOWORD(v75) = sub_3007020(a3, (unsigned int)(v74 + 1));
          LOWORD(v111) = v75;
          v112 = v76;
          if ( !(_WORD)v75 )
          {
            v77 = sub_3007260(&v111);
            v79 = v78;
            v119 = v77;
            v80 = v77;
            v120 = v79;
            goto LABEL_108;
          }
          v75 = (unsigned __int16)v75;
          if ( (_WORD)v75 == 1 || (unsigned __int16)(v75 - 504) <= 7u )
            goto LABEL_178;
          break;
      }
    }
    else
    {
      v112 = 0;
      LOWORD(v111) = 2;
      v75 = 2;
    }
    v79 = 16LL * (v75 - 1);
    v80 = *(_QWORD *)&byte_444C4A0[v79];
    LOBYTE(v79) = byte_444C4A0[v79 + 8];
LABEL_108:
    v115 = v80;
    LOBYTE(v116) = v79;
    v81 = sub_CA1930(&v115);
    if ( v81 <= 8 )
    {
      v82 = 5;
      v108.m128i_i64[1] = 0;
      v108.m128i_i16[0] = 5;
      goto LABEL_110;
    }
    _BitScanReverse(&v81, v81 - 1);
    v84 = v81 ^ 0x1F;
    v85 = 32 - v84;
    if ( v84 == 28 )
    {
      v82 = 6;
      v108.m128i_i64[1] = 0;
      v108.m128i_i16[0] = 6;
      goto LABEL_110;
    }
    if ( v85 == 5 )
    {
      v108.m128i_i16[0] = 7;
      v82 = 7;
      v108.m128i_i64[1] = 0;
      goto LABEL_110;
    }
    if ( v85 != 6 )
      break;
    v108.m128i_i16[0] = 8;
    v82 = 8;
    v108.m128i_i64[1] = 0;
LABEL_110:
    if ( BYTE4(v12) )
      v70 = sub_2D43AD0(v82, v12);
    else
      v70 = sub_2D43050(v82, v12);
    if ( v70 && !*(_BYTE *)(a2 + v70 + 524896) )
    {
      v87 = v108.m128i_i32[0];
      LODWORD(v115) = v12;
      v88 = v108.m128i_i64[1];
      BYTE4(v115) = BYTE4(v12);
      if ( BYTE4(v12) )
        v89 = sub_2D43AD0(v108.m128i_i16[0], v12);
      else
        v89 = sub_2D43050(v108.m128i_i16[0], v12);
      v90 = 0;
      if ( !v89 )
        v89 = sub_3009450(a3, v87, v88, v115);
      *(_BYTE *)a1 = 1;
      *(_WORD *)(a1 + 8) = v89;
      *(_QWORD *)(a1 + 16) = v90;
      return a1;
    }
  }
  if ( v85 == 7 )
  {
    v82 = 9;
    v108.m128i_i64[1] = 0;
    v108.m128i_i16[0] = 9;
    goto LABEL_110;
  }
  v108.m128i_i16[0] = sub_3007020(a3, (unsigned int)(1 << (32 - v84)));
  v82 = v108.m128i_i16[0];
  v108.m128i_i64[1] = v86;
  if ( v108.m128i_i16[0] )
    goto LABEL_110;
  v108.m128i_i16[0] = v15;
  v108.m128i_i64[1] = v100;
LABEL_45:
  while ( 1 )
  {
    v37 = ((((((((((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
               | (unsigned int)v12
               | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 4)
             | (((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
             | (unsigned int)v12
             | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 8)
           | (((((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
             | (unsigned int)v12
             | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 4)
           | (((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
           | (unsigned int)v12
           | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 16)
         | (((((((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
             | (unsigned int)v12
             | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 4)
           | (((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
           | (unsigned int)v12
           | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 8)
         | (((((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
           | (unsigned int)v12
           | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 4)
         | (((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
         | (unsigned int)v12
         | ((unsigned __int64)(unsigned int)v12 >> 1))
        + 1;
    LODWORD(v12) = ((((((((((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
                        | (unsigned int)v12
                        | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 4)
                      | (((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
                      | (unsigned int)v12
                      | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 8)
                    | (((((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
                      | (unsigned int)v12
                      | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 4)
                    | (((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
                    | (unsigned int)v12
                    | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 16)
                  | (((((((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
                      | (unsigned int)v12
                      | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 4)
                    | (((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
                    | (unsigned int)v12
                    | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 8)
                  | (((((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
                    | (unsigned int)v12
                    | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 4)
                  | (((unsigned int)v12 | ((unsigned __int64)(unsigned int)v12 >> 1)) >> 2)
                  | v12
                  | ((unsigned int)v12 >> 1))
                 + 1;
    if ( !v15 )
      break;
    v36 = BYTE4(v12) ? sub_2D43AD0(v15, v37) : sub_2D43050(v15, v37);
    if ( !v36 )
      break;
    if ( !*(_BYTE *)(a2 + v36 + 524896) )
    {
      *(_BYTE *)a1 = 7;
      *(_WORD *)(a1 + 8) = v36;
      *(_QWORD *)(a1 + 16) = 0;
      return a1;
    }
    v15 = v108.m128i_i16[0];
  }
LABEL_36:
  v30 = v104;
  if ( !(_WORD)v104 )
  {
    v51 = sub_3007240(&v104);
    if ( (v51 & (v51 - 1)) == 0 )
    {
      v109 = sub_3007240(&v104);
      if ( (_DWORD)v109 != 1 || !BYTE4(v109) )
      {
        v111 = sub_3007240(&v104);
        v34 = v111;
        v33 = HIDWORD(v111);
        v31 = BYTE4(v111);
        goto LABEL_73;
      }
LABEL_40:
      v35 = _mm_loadu_si128(&v108);
      *(_BYTE *)a1 = 10;
      *(__m128i *)(a1 + 8) = v35;
      return a1;
    }
    v52 = sub_3007240(&v104);
    v53 = v105;
    if ( (v52 & (v52 - 1)) == 0 )
    {
LABEL_82:
      *(_BYTE *)a1 = 7;
      *(_WORD *)(a1 + 8) = v30;
      *(_QWORD *)(a1 + 16) = v53;
      return a1;
    }
    v54 = sub_3007240(&v104);
    v33 = HIDWORD(v54);
    v55 = v54 - 1;
    if ( v55 )
    {
      _BitScanReverse(&v55, v55);
      v56 = 1 << (32 - (v55 ^ 0x1F));
    }
    else
    {
      v56 = 1;
    }
    v57 = sub_3009970(&v104);
    v31 = v33;
    v59 = v58;
LABEL_78:
    v121.m128i_i32[0] = v56;
    v64 = v57;
    v121.m128i_i8[4] = v33;
    if ( v31 )
      v30 = sub_2D43AD0(v57, v56);
    else
      v30 = sub_2D43050(v57, v56);
    v53 = 0;
    if ( !v30 )
      v30 = sub_3009450(a3, v64, v59, v121.m128i_i64[0]);
    goto LABEL_82;
  }
  v31 = (unsigned __int16)(v104 - 176) <= 0x34u;
  v32 = (unsigned __int16)v104 - 1;
  LOBYTE(v33) = v31;
  v34 = word_4456340[v32];
  if ( ((v34 - 1) & v34) != 0 )
  {
    _BitScanReverse(&v63, v34 - 1);
    v59 = 0;
    v56 = 1 << (32 - (v63 ^ 0x1F));
    v57 = word_4456580[v32];
    goto LABEL_78;
  }
  if ( (_WORD)v34 == 1 && (unsigned __int16)(v104 - 176) <= 0x34u )
    goto LABEL_40;
LABEL_73:
  v60 = v108.m128i_i32[0];
  v61 = v34 >> 1;
  v121.m128i_i8[4] = v33;
  v121.m128i_i32[0] = v61;
  v62 = v108.m128i_i64[1];
  if ( v31 )
    v42 = sub_2D43AD0(v108.m128i_i16[0], v61);
  else
    v42 = sub_2D43050(v108.m128i_i16[0], v61);
  v43 = 0;
  if ( !v42 )
    v42 = sub_3009450(a3, v60, v62, v121.m128i_i64[0]);
LABEL_52:
  *(_BYTE *)a1 = 6;
  *(_WORD *)(a1 + 8) = v42;
  *(_QWORD *)(a1 + 16) = v43;
  return a1;
}

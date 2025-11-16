// Function: sub_139E290
// Address: 0x139e290
//
__int64 __fastcall sub_139E290(_QWORD *a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v5; // r12
  __int64 v6; // rax
  size_t v7; // rdx
  _WORD *v8; // rdi
  const char *v9; // rsi
  size_t v10; // r14
  unsigned __int64 v11; // rax
  __int64 result; // rax
  __int64 v13; // rbx
  __int64 v14; // r15
  unsigned __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // r13
  __int64 v18; // r12
  __int64 v19; // r15
  __int64 j; // rbx
  __int64 v21; // r13
  __int64 v22; // r14
  __int64 v23; // r15
  unsigned __int8 v24; // al
  __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // esi
  __int64 v28; // rdi
  __int64 v29; // r8
  int v30; // esi
  unsigned int v31; // ecx
  __int64 *v32; // rdx
  __int64 v33; // r9
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 *v36; // rbx
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 v39; // r12
  __int64 v40; // r15
  _BYTE *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // r8
  _BYTE *v45; // rax
  __m128i *v46; // rdx
  __m128i v47; // xmm0
  __int64 v48; // r8
  __int64 v49; // rax
  size_t v50; // rdx
  __int64 v51; // r8
  const char *v52; // rsi
  _BYTE *v53; // rax
  _BYTE *v54; // rdi
  __m128i *v55; // rdx
  __int64 v56; // r8
  _BYTE *v57; // rax
  __int64 v58; // rdi
  __int64 v59; // rax
  __m128i *v60; // rdx
  __m128i v61; // xmm0
  __int64 v62; // rax
  int v63; // eax
  __int64 v64; // rax
  void *v65; // rdx
  __int64 v66; // r15
  _BYTE *v67; // rax
  __m128i *v68; // rdx
  __m128i v69; // xmm0
  __int64 v70; // rax
  __int64 v71; // rbx
  __int64 v72; // r14
  __int64 v73; // r15
  __int64 v74; // r12
  _BYTE *v75; // rax
  __m128i v76; // xmm0
  __int64 v77; // r12
  _QWORD *v78; // rdx
  _QWORD *v79; // rdx
  _BYTE *v80; // rax
  unsigned int v81; // esi
  __int64 v82; // rbx
  __int64 v83; // r14
  __int64 v84; // r12
  __int64 v85; // r15
  _BYTE *v86; // rax
  int i; // edx
  int v88; // r10d
  __int64 v89; // [rsp+8h] [rbp-D8h]
  __int64 *v90; // [rsp+20h] [rbp-C0h]
  __int64 v91; // [rsp+28h] [rbp-B8h]
  int v92; // [rsp+28h] [rbp-B8h]
  __int64 v93; // [rsp+30h] [rbp-B0h]
  __int64 v94; // [rsp+30h] [rbp-B0h]
  __int64 v95; // [rsp+30h] [rbp-B0h]
  size_t v96; // [rsp+30h] [rbp-B0h]
  int v97; // [rsp+30h] [rbp-B0h]
  __int64 *v98; // [rsp+30h] [rbp-B0h]
  __int64 v100; // [rsp+40h] [rbp-A0h]
  __int64 v101; // [rsp+48h] [rbp-98h]
  _BYTE *v102; // [rsp+50h] [rbp-90h] BYREF
  __int64 v103; // [rsp+58h] [rbp-88h]
  _BYTE v104[32]; // [rsp+60h] [rbp-80h] BYREF
  _BYTE *v105; // [rsp+80h] [rbp-60h] BYREF
  __int64 v106; // [rsp+88h] [rbp-58h]
  _BYTE v107[80]; // [rsp+90h] [rbp-50h] BYREF

  v3 = *(__m128i **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v3 <= 0x1Bu )
  {
    v5 = sub_16E7EE0(a2, "Delinearization on function ", 28);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4289610);
    v5 = a2;
    qmemcpy(&v3[1], "on function ", 12);
    *v3 = si128;
    *(_QWORD *)(a2 + 24) += 28LL;
  }
  v6 = sub_1649960(a1[20]);
  v8 = *(_WORD **)(v5 + 24);
  v9 = (const char *)v6;
  v10 = v7;
  v11 = *(_QWORD *)(v5 + 16) - (_QWORD)v8;
  if ( v11 < v7 )
  {
    v64 = sub_16E7EE0(v5, v9);
    v8 = *(_WORD **)(v64 + 24);
    v5 = v64;
    v11 = *(_QWORD *)(v64 + 16) - (_QWORD)v8;
  }
  else if ( v7 )
  {
    memcpy(v8, v9, v7);
    v8 = (_WORD *)(v10 + *(_QWORD *)(v5 + 24));
    v15 = *(_QWORD *)(v5 + 16) - (_QWORD)v8;
    *(_QWORD *)(v5 + 24) = v8;
    if ( v15 > 1 )
      goto LABEL_6;
LABEL_15:
    sub_16E7EE0(v5, ":\n", 2);
    goto LABEL_7;
  }
  if ( v11 <= 1 )
    goto LABEL_15;
LABEL_6:
  *v8 = 2618;
  *(_QWORD *)(v5 + 24) += 2LL;
LABEL_7:
  result = a1[20];
  v13 = *(_QWORD *)(result + 80);
  v14 = result + 72;
  if ( result + 72 != v13 )
  {
    if ( !v13 )
      BUG();
    while ( 1 )
    {
      result = v13 + 16;
      if ( *(_QWORD *)(v13 + 24) != v13 + 16 )
        break;
      v13 = *(_QWORD *)(v13 + 8);
      if ( v14 == v13 )
        return result;
      if ( !v13 )
        BUG();
    }
    result = a2;
    v16 = v13;
    v17 = *(_QWORD *)(v13 + 24);
    v18 = v14;
    v19 = result;
LABEL_17:
    if ( v16 != v18 )
    {
      j = v17;
      v21 = v16;
      v22 = v19;
      v23 = v18;
      while ( 2 )
      {
        if ( !j )
          BUG();
        v24 = *(_BYTE *)(j - 8);
        if ( (unsigned __int8)(v24 - 54) > 2u )
          goto LABEL_21;
        v26 = a1[21];
        v27 = *(_DWORD *)(v26 + 24);
        if ( !v27 )
          goto LABEL_21;
        v28 = *(_QWORD *)(j + 16);
        v29 = *(_QWORD *)(v26 + 8);
        v30 = v27 - 1;
        v31 = v30 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v32 = (__int64 *)(v29 + 16LL * v31);
        v33 = *v32;
        if ( v28 != *v32 )
        {
          for ( i = 1; ; i = v88 )
          {
            if ( v33 == -8 )
              goto LABEL_21;
            v88 = i + 1;
            v31 = v30 & (i + v31);
            v32 = (__int64 *)(v29 + 16LL * v31);
            v33 = *v32;
            if ( v28 == *v32 )
              break;
          }
        }
        if ( v32[1] )
        {
          v100 = j - 24;
          v34 = 0;
          v101 = j;
          v35 = a1[22];
          v36 = (__int64 *)v32[1];
          v89 = v23;
          if ( v24 > 0x17u )
          {
            if ( v24 == 54 )
              goto LABEL_69;
            goto LABEL_36;
          }
          while ( 1 )
          {
            v37 = sub_1472610(v35, v34, v36);
            v38 = sub_1456F20(a1[22], v37);
            v39 = v38;
            if ( *(_WORD *)(v38 + 24) != 10 )
            {
LABEL_70:
              v23 = v89;
              j = v101;
              break;
            }
            v40 = sub_14806B0(a1[22], v37, v38, 0, 0);
            v41 = *(_BYTE **)(v22 + 24);
            if ( *(_BYTE **)(v22 + 16) == v41 )
            {
              sub_16E7EE0(v22, "\n", 1);
              v42 = *(_QWORD *)(v22 + 24);
              if ( (unsigned __int64)(*(_QWORD *)(v22 + 16) - v42) <= 4 )
              {
LABEL_74:
                v44 = sub_16E7EE0(v22, "Inst:", 5);
                goto LABEL_44;
              }
            }
            else
            {
              *v41 = 10;
              v42 = *(_QWORD *)(v22 + 24) + 1LL;
              v43 = *(_QWORD *)(v22 + 16);
              *(_QWORD *)(v22 + 24) = v42;
              if ( (unsigned __int64)(v43 - v42) <= 4 )
                goto LABEL_74;
            }
            *(_DWORD *)v42 = 1953721929;
            v44 = v22;
            *(_BYTE *)(v42 + 4) = 58;
            *(_QWORD *)(v22 + 24) += 5LL;
LABEL_44:
            v93 = v44;
            sub_155C2B0(v100, v44, 0);
            v45 = *(_BYTE **)(v93 + 24);
            if ( *(_BYTE **)(v93 + 16) == v45 )
            {
              sub_16E7EE0(v93, "\n", 1);
            }
            else
            {
              *v45 = 10;
              ++*(_QWORD *)(v93 + 24);
            }
            v46 = *(__m128i **)(v22 + 24);
            if ( *(_QWORD *)(v22 + 16) - (_QWORD)v46 <= 0x14u )
            {
              v48 = sub_16E7EE0(v22, "In Loop with Header: ", 21);
            }
            else
            {
              v47 = _mm_load_si128((const __m128i *)&xmmword_4289620);
              v46[1].m128i_i32[0] = 980575588;
              v48 = v22;
              v46[1].m128i_i8[4] = 32;
              *v46 = v47;
              *(_QWORD *)(v22 + 24) += 21LL;
            }
            v94 = v48;
            v49 = sub_1649960(*(_QWORD *)v36[4]);
            v51 = v94;
            v52 = (const char *)v49;
            v53 = *(_BYTE **)(v94 + 16);
            v54 = *(_BYTE **)(v94 + 24);
            if ( v53 - v54 < v50 )
            {
              v51 = sub_16E7EE0(v94, v52);
              v53 = *(_BYTE **)(v51 + 16);
              v54 = *(_BYTE **)(v51 + 24);
            }
            else if ( v50 )
            {
              v91 = v94;
              v96 = v50;
              memcpy(v54, v52, v50);
              v51 = v91;
              v54 = (_BYTE *)(*(_QWORD *)(v91 + 24) + v96);
              v53 = *(_BYTE **)(v91 + 16);
              *(_QWORD *)(v91 + 24) = v54;
            }
            if ( v54 == v53 )
            {
              sub_16E7EE0(v51, "\n", 1);
            }
            else
            {
              *v54 = 10;
              ++*(_QWORD *)(v51 + 24);
            }
            v55 = *(__m128i **)(v22 + 24);
            if ( *(_QWORD *)(v22 + 16) - (_QWORD)v55 <= 0xFu )
            {
              v56 = sub_16E7EE0(v22, "AccessFunction: ", 16);
            }
            else
            {
              v56 = v22;
              *v55 = _mm_load_si128((const __m128i *)&xmmword_4289630);
              *(_QWORD *)(v22 + 24) += 16LL;
            }
            v95 = v56;
            sub_1456620(v40, v56);
            v57 = *(_BYTE **)(v95 + 24);
            if ( *(_BYTE **)(v95 + 16) == v57 )
            {
              sub_16E7EE0(v95, "\n", 1);
            }
            else
            {
              *v57 = 10;
              ++*(_QWORD *)(v95 + 24);
            }
            v58 = a1[22];
            v102 = v104;
            v105 = v107;
            v103 = 0x300000000LL;
            v106 = 0x300000000LL;
            v59 = sub_145D1F0(v58, v100);
            sub_1490760(v58, v40, &v102, &v105, v59);
            if ( (_DWORD)v103 && (unsigned int)v103 == (unsigned __int64)(unsigned int)v106 && (_DWORD)v106 )
            {
              v65 = *(void **)(v22 + 24);
              if ( *(_QWORD *)(v22 + 16) - (_QWORD)v65 <= 0xCu )
              {
                v66 = sub_16E7EE0(v22, "Base offset: ", 13);
              }
              else
              {
                v66 = v22;
                qmemcpy(v65, "Base offset: ", 13);
                *(_QWORD *)(v22 + 24) += 13LL;
              }
              sub_1456620(v39, v66);
              v67 = *(_BYTE **)(v66 + 24);
              if ( *(_BYTE **)(v66 + 16) == v67 )
              {
                sub_16E7EE0(v66, "\n", 1);
              }
              else
              {
                *v67 = 10;
                ++*(_QWORD *)(v66 + 24);
              }
              v68 = *(__m128i **)(v22 + 24);
              if ( *(_QWORD *)(v22 + 16) - (_QWORD)v68 <= 0x15u )
              {
                sub_16E7EE0(v22, "ArrayDecl[UnknownSize]", 22);
                v70 = *(_QWORD *)(v22 + 24);
              }
              else
              {
                v69 = _mm_load_si128((const __m128i *)&xmmword_4289640);
                v68[1].m128i_i32[0] = 2053722990;
                v68[1].m128i_i16[2] = 23909;
                *v68 = v69;
                v70 = *(_QWORD *)(v22 + 24) + 22LL;
                *(_QWORD *)(v22 + 24) = v70;
              }
              v97 = v103;
              v92 = v103 - 1;
              if ( (int)v103 - 1 > 0 )
              {
                v90 = v36;
                v71 = v22;
                v72 = 0;
                v73 = 8LL * (unsigned int)(v103 - 2) + 8;
                do
                {
                  while ( 1 )
                  {
                    if ( *(_QWORD *)(v71 + 16) == v70 )
                    {
                      v74 = sub_16E7EE0(v71, "[", 1);
                    }
                    else
                    {
                      *(_BYTE *)v70 = 91;
                      v74 = v71;
                      ++*(_QWORD *)(v71 + 24);
                    }
                    sub_1456620(*(_QWORD *)&v105[v72], v74);
                    v75 = *(_BYTE **)(v74 + 24);
                    if ( *(_BYTE **)(v74 + 16) == v75 )
                      break;
                    v72 += 8;
                    *v75 = 93;
                    ++*(_QWORD *)(v74 + 24);
                    v70 = *(_QWORD *)(v71 + 24);
                    if ( v73 == v72 )
                      goto LABEL_104;
                  }
                  v72 += 8;
                  sub_16E7EE0(v74, "]", 1);
                  v70 = *(_QWORD *)(v71 + 24);
                }
                while ( v73 != v72 );
LABEL_104:
                v22 = v71;
                v36 = v90;
              }
              if ( (unsigned __int64)(*(_QWORD *)(v22 + 16) - v70) <= 0x11 )
              {
                v77 = sub_16E7EE0(v22, " with elements of ", 18);
              }
              else
              {
                v76 = _mm_load_si128((const __m128i *)&xmmword_4289660);
                v77 = v22;
                *(_WORD *)(v70 + 16) = 8294;
                *(__m128i *)v70 = v76;
                *(_QWORD *)(v22 + 24) += 18LL;
              }
              sub_1456620(*(_QWORD *)&v105[8 * v92], v77);
              v78 = *(_QWORD **)(v77 + 24);
              if ( *(_QWORD *)(v77 + 16) - (_QWORD)v78 <= 7u )
              {
                sub_16E7EE0(v77, " bytes.\n", 8);
              }
              else
              {
                *v78 = 0xA2E736574796220LL;
                *(_QWORD *)(v77 + 24) += 8LL;
              }
              v79 = *(_QWORD **)(v22 + 24);
              if ( *(_QWORD *)(v22 + 16) - (_QWORD)v79 <= 7u )
              {
                sub_16E7EE0(v22, "ArrayRef", 8);
                v80 = *(_BYTE **)(v22 + 24);
              }
              else
              {
                *v79 = 0x6665527961727241LL;
                v80 = (_BYTE *)(*(_QWORD *)(v22 + 24) + 8LL);
                *(_QWORD *)(v22 + 24) = v80;
              }
              v81 = v97;
              if ( v97 > 0 )
              {
                v98 = v36;
                v82 = v22;
                v83 = 0;
                v84 = 8LL * v81;
                do
                {
                  while ( 1 )
                  {
                    if ( *(_BYTE **)(v82 + 16) == v80 )
                    {
                      v85 = sub_16E7EE0(v82, "[", 1);
                    }
                    else
                    {
                      *v80 = 91;
                      v85 = v82;
                      ++*(_QWORD *)(v82 + 24);
                    }
                    sub_1456620(*(_QWORD *)&v102[v83], v85);
                    v86 = *(_BYTE **)(v85 + 24);
                    if ( *(_BYTE **)(v85 + 16) == v86 )
                      break;
                    v83 += 8;
                    *v86 = 93;
                    ++*(_QWORD *)(v85 + 24);
                    v80 = *(_BYTE **)(v82 + 24);
                    if ( v84 == v83 )
                      goto LABEL_119;
                  }
                  v83 += 8;
                  sub_16E7EE0(v85, "]", 1);
                  v80 = *(_BYTE **)(v82 + 24);
                }
                while ( v84 != v83 );
LABEL_119:
                v22 = v82;
                v36 = v98;
              }
              if ( v80 == *(_BYTE **)(v22 + 16) )
              {
                sub_16E7EE0(v22, "\n", 1);
              }
              else
              {
                *v80 = 10;
                ++*(_QWORD *)(v22 + 24);
              }
            }
            else
            {
              v60 = *(__m128i **)(v22 + 24);
              if ( *(_QWORD *)(v22 + 16) - (_QWORD)v60 <= 0x15u )
              {
                sub_16E7EE0(v22, "failed to delinearize\n", 22);
              }
              else
              {
                v61 = _mm_load_si128((const __m128i *)&xmmword_4289650);
                v60[1].m128i_i32[0] = 2053730913;
                v60[1].m128i_i16[2] = 2661;
                *v60 = v61;
                *(_QWORD *)(v22 + 24) += 22LL;
              }
            }
            if ( v105 != v107 )
              _libc_free((unsigned __int64)v105);
            if ( v102 != v104 )
              _libc_free((unsigned __int64)v102);
            v36 = (__int64 *)*v36;
            if ( !v36 )
              goto LABEL_70;
            v35 = a1[22];
            v34 = 0;
            v24 = *(_BYTE *)(v101 - 8);
            if ( v24 > 0x17u )
            {
              if ( v24 == 54 )
              {
LABEL_69:
                v34 = *(_QWORD *)(v101 - 48);
              }
              else
              {
LABEL_36:
                switch ( v24 )
                {
                  case '7':
                    goto LABEL_69;
                  case 'N':
                    v62 = *(_QWORD *)(v101 - 48);
                    if ( !*(_BYTE *)(v62 + 16) )
                    {
                      v63 = *(_DWORD *)(v62 + 36);
                      if ( v63 == 4085 || v63 == 4057 )
                      {
                        v34 = *(_QWORD *)(v101 + 24 * (1LL - (*(_DWORD *)(v101 - 4) & 0xFFFFFFF)) - 24);
                      }
                      else if ( v63 == 4503 || v63 == 4492 )
                      {
                        v34 = *(_QWORD *)(v101 + 24 * (2LL - (*(_DWORD *)(v101 - 4) & 0xFFFFFFF)) - 24);
                      }
                    }
                    break;
                  case '8':
                    v34 = *(_QWORD *)(v100 - 24LL * (*(_DWORD *)(v101 - 4) & 0xFFFFFFF));
                    break;
                }
              }
            }
          }
        }
LABEL_21:
        for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v21 + 24) )
        {
          v25 = v21 - 24;
          if ( !v21 )
            v25 = 0;
          result = v25 + 40;
          if ( j != result )
            break;
          v21 = *(_QWORD *)(v21 + 8);
          if ( v23 == v21 )
          {
            v18 = v23;
            v19 = v22;
            v16 = v21;
            v17 = j;
            goto LABEL_17;
          }
          if ( !v21 )
            BUG();
        }
        if ( v23 != v21 )
          continue;
        break;
      }
    }
  }
  return result;
}

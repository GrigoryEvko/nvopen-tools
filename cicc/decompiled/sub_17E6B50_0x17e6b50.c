// Function: sub_17E6B50
// Address: 0x17e6b50
//
_BYTE *__fastcall sub_17E6B50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  _BYTE *v6; // r14
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  _BYTE *result; // rax
  __int64 **v12; // r12
  unsigned int i; // r14d
  _BYTE *v14; // rdx
  char *v15; // rcx
  char *v16; // rsi
  bool v17; // zf
  char *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rax
  _WORD *v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // rsi
  unsigned int v28; // ecx
  __int64 *v29; // rax
  __int64 v30; // r11
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // r8
  __int64 v36; // rsi
  unsigned int v37; // ecx
  __int64 *v38; // rax
  __int64 v39; // r11
  __int64 v40; // r14
  __int64 *v41; // rax
  size_t v42; // rdx
  char *v43; // rcx
  char *v44; // rsi
  char *v45; // rax
  char v46; // al
  const char **v47; // rdx
  char v48; // al
  __m128i *v49; // rcx
  char v50; // dl
  _QWORD *v51; // rsi
  int v52; // eax
  int v53; // eax
  __int64 *v54; // rax
  __int64 *v55; // r12
  __int64 v56; // r8
  __int64 *v57; // r13
  __int64 v58; // rdx
  __int64 v59; // r14
  const char *v60; // rax
  size_t v61; // rdx
  _WORD *v62; // rdi
  char *v63; // rsi
  unsigned __int64 v64; // rax
  __int64 v65; // rax
  int v66; // eax
  __int64 v67; // rdi
  _BYTE *v68; // rax
  int v69; // eax
  char v70; // al
  _QWORD *v71; // rdx
  char v72; // al
  __m128i *v73; // rsi
  char v74; // dl
  __m128i *v75; // rdi
  _WORD *v76; // rdx
  unsigned __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  int v80; // r10d
  int v81; // r10d
  __int64 **v82; // [rsp+10h] [rbp-140h]
  size_t v83; // [rsp+18h] [rbp-138h]
  __int64 v84; // [rsp+18h] [rbp-138h]
  unsigned int v86; // [rsp+30h] [rbp-120h]
  const char *v87; // [rsp+40h] [rbp-110h] BYREF
  char v88; // [rsp+50h] [rbp-100h]
  char v89; // [rsp+51h] [rbp-FFh]
  _QWORD v90[2]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v91; // [rsp+70h] [rbp-E0h]
  __m128i v92; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v93; // [rsp+90h] [rbp-C0h]
  __m128i v94; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v95; // [rsp+B0h] [rbp-A0h]
  __m128i v96; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v97; // [rsp+D0h] [rbp-80h]
  __m128i *v98; // [rsp+E0h] [rbp-70h] BYREF
  size_t v99; // [rsp+E8h] [rbp-68h]
  _QWORD v100[2]; // [rsp+F0h] [rbp-60h] BYREF
  __m128i *v101; // [rsp+100h] [rbp-50h] BYREF
  _BYTE *v102; // [rsp+108h] [rbp-48h]
  _QWORD v103[8]; // [rsp+110h] [rbp-40h] BYREF

  v3 = a1;
  sub_16E2FC0((__int64 *)&v101, a3);
  v6 = v102;
  if ( v101 != (__m128i *)v103 )
    j_j___libc_free_0(v101, v103[0] + 1LL);
  if ( v6 )
  {
    sub_16E2CE0(a3, a2);
    sub_1263B40(a2, "\n");
  }
  v7 = sub_1263B40(a2, "  Number of Basic Blocks: ");
  v8 = sub_16E7A90(v7, *(unsigned int *)(a1 + 48));
  sub_1263B40(v8, "\n");
  if ( *(_DWORD *)(a1 + 48) )
  {
    v54 = *(__int64 **)(a1 + 40);
    v55 = &v54[2 * *(unsigned int *)(a1 + 56)];
    if ( v54 != v55 )
    {
      while ( 1 )
      {
        v56 = *v54;
        if ( *v54 != -16 && v56 != -8 )
          break;
        v54 += 2;
        if ( v55 == v54 )
          goto LABEL_6;
      }
      if ( v55 != v54 )
      {
        v57 = v54;
        while ( 1 )
        {
          v58 = *(_QWORD *)(a2 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v58) <= 5 )
          {
            v84 = v56;
            v78 = sub_16E7EE0(a2, "  BB: ", 6u);
            v56 = v84;
            v59 = v78;
          }
          else
          {
            *(_DWORD *)v58 = 1111629856;
            v59 = a2;
            *(_WORD *)(v58 + 4) = 8250;
            *(_QWORD *)(a2 + 24) += 6LL;
          }
          if ( v56 )
          {
            v60 = sub_1649960(v56);
            v62 = *(_WORD **)(v59 + 24);
            v63 = (char *)v60;
            v64 = *(_QWORD *)(v59 + 16) - (_QWORD)v62;
            if ( v61 > v64 )
              goto LABEL_114;
            if ( !v61 )
              goto LABEL_78;
          }
          else
          {
            v62 = *(_WORD **)(v59 + 24);
            v61 = 8;
            v63 = "FakeNode";
            if ( *(_QWORD *)(v59 + 16) - (_QWORD)v62 <= 7u )
            {
LABEL_114:
              v79 = sub_16E7EE0(v59, v63, v61);
              v62 = *(_WORD **)(v79 + 24);
              v59 = v79;
              v64 = *(_QWORD *)(v79 + 16) - (_QWORD)v62;
LABEL_78:
              if ( v64 > 1 )
                goto LABEL_79;
              goto LABEL_109;
            }
          }
          v83 = v61;
          memcpy(v62, v63, v61);
          v76 = (_WORD *)(*(_QWORD *)(v59 + 24) + v83);
          v77 = *(_QWORD *)(v59 + 16) - (_QWORD)v76;
          *(_QWORD *)(v59 + 24) = v76;
          v62 = v76;
          if ( v77 > 1 )
          {
LABEL_79:
            *v62 = 8224;
            *(_QWORD *)(v59 + 24) += 2LL;
            goto LABEL_80;
          }
LABEL_109:
          v59 = sub_16E7EE0(v59, "  ", 2u);
LABEL_80:
          v65 = v57[1];
          if ( !*(_BYTE *)(v65 + 24) )
          {
            v66 = *(_DWORD *)(v65 + 8);
            LOWORD(v97) = 2307;
            LODWORD(v101) = v66;
            v96.m128i_i64[0] = (__int64)"Index=";
            v96.m128i_i64[1] = (__int64)v101;
            sub_16E2FC0((__int64 *)&v98, (__int64)&v96);
            goto LABEL_82;
          }
          v94.m128i_i64[0] = v65 + 16;
          LOWORD(v95) = 267;
          v90[0] = "  Count=";
          v91 = 259;
          v69 = *(_DWORD *)(v65 + 8);
          LOWORD(v97) = 2307;
          LODWORD(v98) = v69;
          v96.m128i_i64[0] = (__int64)"Index=";
          v96.m128i_i64[1] = (__int64)v98;
          sub_16E2FC0((__int64 *)&v101, (__int64)&v96);
          v70 = v91;
          if ( (_BYTE)v91 )
          {
            if ( (_BYTE)v91 == 1 )
            {
              LOWORD(v93) = 260;
              v92.m128i_i64[0] = (__int64)&v101;
            }
            else
            {
              v71 = (_QWORD *)v90[0];
              if ( HIBYTE(v91) != 1 )
              {
                v71 = v90;
                v70 = 2;
              }
              v92.m128i_i64[1] = (__int64)v71;
              LOBYTE(v93) = 4;
              v92.m128i_i64[0] = (__int64)&v101;
              BYTE1(v93) = v70;
            }
            v72 = v95;
            if ( (_BYTE)v95 )
            {
              if ( (_BYTE)v95 == 1 )
              {
                v96 = _mm_loadu_si128(&v92);
                v97 = v93;
              }
              else
              {
                v73 = &v92;
                v74 = 2;
                if ( BYTE1(v93) == 1 )
                {
                  v73 = (__m128i *)v92.m128i_i64[0];
                  v74 = 4;
                }
                v75 = (__m128i *)v94.m128i_i64[0];
                if ( BYTE1(v95) != 1 )
                {
                  v75 = &v94;
                  v72 = 2;
                }
                v96.m128i_i64[0] = (__int64)v73;
                v96.m128i_i64[1] = (__int64)v75;
                LOBYTE(v97) = v74;
                BYTE1(v97) = v72;
              }
              goto LABEL_105;
            }
          }
          else
          {
            LOWORD(v93) = 256;
          }
          LOWORD(v97) = 256;
LABEL_105:
          sub_16E2FC0((__int64 *)&v98, (__int64)&v96);
          if ( v101 != (__m128i *)v103 )
            j_j___libc_free_0(v101, v103[0] + 1LL);
LABEL_82:
          v67 = sub_16E7EE0(v59, v98->m128i_i8, v99);
          v68 = *(_BYTE **)(v67 + 24);
          if ( *(_BYTE **)(v67 + 16) == v68 )
          {
            sub_16E7EE0(v67, "\n", 1u);
          }
          else
          {
            *v68 = 10;
            ++*(_QWORD *)(v67 + 24);
          }
          if ( v98 != (__m128i *)v100 )
            j_j___libc_free_0(v98, v100[0] + 1LL);
          v57 += 2;
          if ( v57 != v55 )
          {
            while ( 1 )
            {
              v56 = *v57;
              if ( *v57 != -8 && v56 != -16 )
                break;
              v57 += 2;
              if ( v55 == v57 )
                goto LABEL_90;
            }
            if ( v55 != v57 )
              continue;
          }
LABEL_90:
          v3 = a1;
          break;
        }
      }
    }
  }
LABEL_6:
  v9 = sub_1263B40(a2, "  Number of Edges: ");
  v10 = sub_16E7A90(v9, (__int64)(*(_QWORD *)(v3 + 16) - *(_QWORD *)(v3 + 8)) >> 3);
  sub_1263B40(v10, " (*: Instrument, C: CriticalEdge, -: Removed)\n");
  result = *(_BYTE **)(v3 + 16);
  v12 = *(__int64 ***)(v3 + 8);
  v82 = (__int64 **)result;
  if ( result != (_BYTE *)v12 )
  {
    for ( i = 0; ; i = v86 )
    {
      v20 = *(_QWORD *)(a2 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v20) <= 6 )
      {
        v21 = sub_16E7EE0(a2, "  Edge ", 7u);
      }
      else
      {
        *(_DWORD *)v20 = 1682251808;
        v21 = a2;
        *(_WORD *)(v20 + 4) = 25959;
        *(_BYTE *)(v20 + 6) = 32;
        *(_QWORD *)(a2 + 24) += 7LL;
      }
      v86 = i + 1;
      v22 = sub_16E7A90(v21, i);
      v23 = *(_WORD **)(v22 + 24);
      v24 = v22;
      if ( *(_QWORD *)(v22 + 16) - (_QWORD)v23 <= 1u )
      {
        v24 = sub_16E7EE0(v22, ": ", 2u);
      }
      else
      {
        *v23 = 8250;
        *(_QWORD *)(v22 + 24) += 2LL;
      }
      v25 = *(unsigned int *)(v3 + 56);
      v26 = *(_QWORD *)(v3 + 40);
      if ( (_DWORD)v25 )
      {
        v27 = **v12;
        v28 = (v25 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v29 = (__int64 *)(v26 + 16LL * v28);
        v30 = *v29;
        if ( v27 == *v29 )
          goto LABEL_27;
        v53 = 1;
        while ( v30 != -8 )
        {
          v80 = v53 + 1;
          v28 = (v25 - 1) & (v53 + v28);
          v29 = (__int64 *)(v26 + 16LL * v28);
          v30 = *v29;
          if ( v27 == *v29 )
            goto LABEL_27;
          v53 = v80;
        }
      }
      v29 = (__int64 *)(v26 + 16 * v25);
LABEL_27:
      v31 = sub_16E7A90(v24, *(unsigned int *)(v29[1] + 8));
      v32 = *(_QWORD *)(v31 + 24);
      v33 = v31;
      if ( (unsigned __int64)(*(_QWORD *)(v31 + 16) - v32) <= 2 )
      {
        v33 = sub_16E7EE0(v31, "-->", 3u);
      }
      else
      {
        *(_BYTE *)(v32 + 2) = 62;
        *(_WORD *)v32 = 11565;
        *(_QWORD *)(v31 + 24) += 3LL;
      }
      v34 = *(unsigned int *)(v3 + 56);
      v35 = *(_QWORD *)(v3 + 40);
      if ( (_DWORD)v34 )
      {
        v36 = (*v12)[1];
        v37 = (v34 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
        v38 = (__int64 *)(v35 + 16LL * v37);
        v39 = *v38;
        if ( v36 == *v38 )
          goto LABEL_31;
        v52 = 1;
        while ( v39 != -8 )
        {
          v81 = v52 + 1;
          v37 = (v34 - 1) & (v52 + v37);
          v38 = (__int64 *)(v35 + 16LL * v37);
          v39 = *v38;
          if ( v36 == *v38 )
            goto LABEL_31;
          v52 = v81;
        }
      }
      v38 = (__int64 *)(v35 + 16 * v34);
LABEL_31:
      v40 = sub_16E7A90(v33, *(unsigned int *)(v38[1] + 8));
      v41 = *v12;
      if ( *((_BYTE *)*v12 + 27) )
      {
        v91 = 267;
        v90[0] = v41 + 4;
        v17 = *((_BYTE *)v41 + 26) == 0;
        v42 = (size_t)(v41 + 2);
        v87 = "  Count=";
        v43 = " ";
        if ( !v17 )
          v43 = (char *)"c";
        v17 = *((_BYTE *)v41 + 24) == 0;
        v44 = "*";
        v89 = 1;
        if ( !v17 )
          v44 = " ";
        v17 = *((_BYTE *)v41 + 25) == 0;
        v88 = 3;
        v45 = "-";
        v94.m128i_i64[1] = (__int64)v43;
        if ( v17 )
          v45 = " ";
        LOWORD(v97) = 770;
        LOWORD(v93) = 771;
        v92.m128i_i64[0] = (__int64)v45;
        v92.m128i_i64[1] = (__int64)v44;
        v94.m128i_i64[0] = (__int64)&v92;
        LOWORD(v95) = 770;
        v96.m128i_i64[0] = (__int64)&v94;
        v96.m128i_i64[1] = (__int64)"  W=";
        v99 = v42;
        v98 = &v96;
        LOWORD(v100[0]) = 2818;
        sub_16E2FC0((__int64 *)&v101, (__int64)&v98);
        v46 = v88;
        if ( v88 )
        {
          if ( v88 == 1 )
          {
            LOWORD(v95) = 260;
            v94.m128i_i64[0] = (__int64)&v101;
          }
          else
          {
            v47 = (const char **)v87;
            if ( v89 != 1 )
            {
              v47 = &v87;
              v46 = 2;
            }
            v94.m128i_i64[1] = (__int64)v47;
            LOBYTE(v95) = 4;
            v94.m128i_i64[0] = (__int64)&v101;
            BYTE1(v95) = v46;
          }
          v48 = v91;
          if ( (_BYTE)v91 )
          {
            if ( (_BYTE)v91 == 1 )
            {
              v96 = _mm_loadu_si128(&v94);
              v97 = v95;
            }
            else
            {
              v49 = &v94;
              v50 = 2;
              if ( BYTE1(v95) == 1 )
              {
                v49 = (__m128i *)v94.m128i_i64[0];
                v50 = 4;
              }
              v51 = (_QWORD *)v90[0];
              if ( HIBYTE(v91) != 1 )
              {
                v51 = v90;
                v48 = 2;
              }
              v96.m128i_i64[0] = (__int64)v49;
              v96.m128i_i64[1] = (__int64)v51;
              LOBYTE(v97) = v50;
              BYTE1(v97) = v48;
            }
            goto LABEL_58;
          }
        }
        else
        {
          LOWORD(v95) = 256;
        }
        LOWORD(v97) = 256;
LABEL_58:
        sub_16E2FC0((__int64 *)&v98, (__int64)&v96);
        if ( v101 != (__m128i *)v103 )
          j_j___libc_free_0(v101, v103[0] + 1LL);
        goto LABEL_15;
      }
      v14 = v41 + 2;
      v15 = " ";
      if ( *((_BYTE *)v41 + 26) )
        v15 = (char *)"c";
      v16 = "*";
      if ( *((_BYTE *)v41 + 24) )
        v16 = " ";
      v17 = *((_BYTE *)v41 + 25) == 0;
      LOWORD(v93) = 771;
      v18 = "-";
      v94.m128i_i64[1] = (__int64)v15;
      if ( v17 )
        v18 = " ";
      v92.m128i_i64[1] = (__int64)v16;
      v102 = v14;
      v92.m128i_i64[0] = (__int64)v18;
      v94.m128i_i64[0] = (__int64)&v92;
      LOWORD(v95) = 770;
      v96.m128i_i64[0] = (__int64)&v94;
      v96.m128i_i64[1] = (__int64)"  W=";
      LOWORD(v97) = 770;
      v101 = &v96;
      LOWORD(v103[0]) = 2818;
      sub_16E2FC0((__int64 *)&v98, (__int64)&v101);
LABEL_15:
      v19 = sub_16E7EE0(v40, v98->m128i_i8, v99);
      result = *(_BYTE **)(v19 + 24);
      if ( *(_BYTE **)(v19 + 16) == result )
      {
        result = (_BYTE *)sub_16E7EE0(v19, "\n", 1u);
      }
      else
      {
        *result = 10;
        ++*(_QWORD *)(v19 + 24);
      }
      if ( v98 != (__m128i *)v100 )
        result = (_BYTE *)j_j___libc_free_0(v98, v100[0] + 1LL);
      if ( v82 == ++v12 )
        return result;
    }
  }
  return result;
}

// Function: sub_3970E40
// Address: 0x3970e40
//
void __fastcall sub_3970E40(__int64 a1, __int64 a2)
{
  void (*v4)(void); // rax
  __int64 **v5; // rbx
  __int64 **v6; // r14
  void (*v7)(); // rax
  __int64 *v8; // rdi
  __int64 v9; // rax
  void (*v10)(void); // rdx
  unsigned int v11; // esi
  __int64 v12; // rdi
  void (*v13)(); // rax
  __int64 v14; // r14
  __int64 v15; // rdi
  void (*v16)(); // rax
  __int64 v17; // r14
  __int64 v18; // rdx
  int v19; // eax
  __int64 v20; // rsi
  int v21; // ecx
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 *v25; // r15
  __int64 *v26; // r14
  __int64 *v27; // rcx
  __int64 v28; // rax
  void (*v29)(); // rbx
  _QWORD *v30; // rax
  __int32 v31; // edx
  __int64 v32; // rax
  char v33; // al
  char **v34; // rcx
  char v35; // dl
  __m128i *v36; // rsi
  char v37; // al
  _QWORD *v38; // rcx
  char v39; // dl
  __m128i *v40; // rsi
  __m128i *v41; // rcx
  char v42; // dl
  __m128i *v43; // rsi
  __m128i *v44; // rcx
  _QWORD *v45; // rax
  __int32 v46; // edx
  _BYTE *v47; // rax
  unsigned int v48; // eax
  __int64 v49; // r14
  void (__fastcall *v50)(__int64, __int64, _QWORD); // rbx
  __int64 v51; // rax
  __int64 v52; // rdx
  int v53; // eax
  __int64 v54; // rsi
  int v55; // ecx
  unsigned int v56; // edx
  __int64 *v57; // rax
  __int64 v58; // rdi
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // rdi
  unsigned __int64 v62; // rax
  __int64 v63; // r13
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // r15
  unsigned int v67; // ebx
  int v68; // r13d
  _BYTE *v69; // rdi
  __int64 v70; // rax
  unsigned __int64 v71; // rdx
  __int64 v72; // rdi
  void (__fastcall *v73)(__int64, __m128i *, _QWORD); // rax
  __int64 *v74; // rax
  __int64 v75; // rdx
  __int64 *v76; // r14
  __int64 *i; // rbx
  __int64 v78; // rsi
  _QWORD *v79; // r15
  __int64 v80; // rsi
  __int64 v81; // rdi
  _BYTE *v82; // rax
  __int64 v83; // r14
  unsigned int v84; // eax
  _WORD *v85; // rdx
  _QWORD *v86; // rax
  int v87; // edx
  int v88; // esi
  unsigned int v89; // esi
  __int64 v90; // rdx
  _BYTE *v91; // rdi
  __int64 v92; // rax
  __int64 v93; // rdx
  __int64 *v94; // rdi
  __m128i v95; // xmm2
  __m128i v96; // xmm0
  int v97; // eax
  int v98; // r8d
  __int64 v99; // rdx
  int v100; // eax
  int v101; // r8d
  bool v102[3]; // [rsp+Dh] [rbp-173h] BYREF
  __int64 v103; // [rsp+10h] [rbp-170h]
  __m128i v104; // [rsp+30h] [rbp-150h] BYREF
  __int64 v105; // [rsp+40h] [rbp-140h]
  char *v106; // [rsp+50h] [rbp-130h] BYREF
  char v107; // [rsp+60h] [rbp-120h]
  char v108; // [rsp+61h] [rbp-11Fh]
  __m128i v109; // [rsp+70h] [rbp-110h] BYREF
  __int64 v110; // [rsp+80h] [rbp-100h]
  _QWORD *v111; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v112; // [rsp+A0h] [rbp-E0h]
  __m128i v113; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v114; // [rsp+C0h] [rbp-C0h]
  __m128i v115; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v116; // [rsp+E0h] [rbp-A0h]
  __m128i v117; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v118; // [rsp+100h] [rbp-80h]
  __m128i v119; // [rsp+110h] [rbp-70h] BYREF
  __int64 v120; // [rsp+120h] [rbp-60h]
  __m128i v121; // [rsp+130h] [rbp-50h] BYREF
  __int64 v122; // [rsp+140h] [rbp-40h]

  v4 = *(void (**)(void))(**(_QWORD **)(a1 + 256) + 144LL);
  if ( v4 != nullsub_581 )
    v4();
  if ( *(_BYTE *)(a2 + 183) )
  {
    v5 = *(__int64 ***)(a1 + 424);
    v6 = &v5[5 * *(unsigned int *)(a1 + 432)];
    if ( v5 != v6 )
    {
      while ( 1 )
      {
        v8 = *v5;
        v9 = **v5;
        v10 = *(void (**)(void))(v9 + 80);
        if ( v10 == nullsub_785 )
        {
          v7 = *(void (**)())(v9 + 72);
          if ( v7 == nullsub_784 )
            goto LABEL_7;
LABEL_10:
          v5 += 5;
          ((void (__fastcall *)(__int64 *, __int64, _QWORD))v7)(v8, a2, 0);
          if ( v6 == v5 )
            break;
        }
        else
        {
          v10();
          v8 = *v5;
          v7 = *(void (**)())(**v5 + 72);
          if ( v7 != nullsub_784 )
            goto LABEL_10;
LABEL_7:
          v5 += 5;
          if ( v6 == v5 )
            break;
        }
      }
    }
  }
  v11 = *(_DWORD *)(a2 + 176);
  if ( v11 )
    sub_396F480(a1, v11, 0);
  sub_3970C70((__int64 *)a1, (_QWORD *)a2, v102);
  v12 = *(_QWORD *)(a1 + 256);
  v13 = *(void (**)())(*(_QWORD *)v12 + 536LL);
  if ( v13 != nullsub_587 )
    ((void (__fastcall *)(__int64, bool *))v13)(v12, v102);
  if ( *(_BYTE *)(a2 + 181) )
  {
    v14 = *(_QWORD *)(a2 + 40);
    if ( *(_BYTE *)(a1 + 416) )
    {
      v15 = *(_QWORD *)(a1 + 256);
      v16 = *(void (**)())(*(_QWORD *)v15 + 104LL);
      v121.m128i_i64[0] = (__int64)"Block address taken";
      LOWORD(v122) = 259;
      if ( v16 != nullsub_580 )
        ((void (__fastcall *)(__int64, __m128i *, __int64))v16)(v15, &v121, 1);
    }
    if ( *(_WORD *)(v14 + 18) )
    {
      v74 = sub_1E2EA70(*(_QWORD *)(a1 + 272), v14);
      v76 = &v74[v75];
      for ( i = v74; v76 != i; ++i )
      {
        v78 = *i;
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 176LL))(
          *(_QWORD *)(a1 + 256),
          v78,
          0);
      }
    }
  }
  if ( !*(_BYTE *)(a1 + 416) )
  {
    if ( *(_QWORD *)(a2 + 64) == *(_QWORD *)(a2 + 72) )
      return;
    goto LABEL_59;
  }
  v17 = *(_QWORD *)(a2 + 40);
  if ( v17 && (*(_BYTE *)(v17 + 23) & 0x20) != 0 )
  {
    v79 = (_QWORD *)sub_157EB90(*(_QWORD *)(a2 + 40));
    v80 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 256) + 112LL))(*(_QWORD *)(a1 + 256));
    sub_15537D0(v17, v80, 0, v79);
    v81 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 256) + 112LL))(*(_QWORD *)(a1 + 256));
    v82 = *(_BYTE **)(v81 + 24);
    if ( (unsigned __int64)v82 >= *(_QWORD *)(v81 + 16) )
    {
      sub_16E7DE0(v81, 10);
    }
    else
    {
      *(_QWORD *)(v81 + 24) = v82 + 1;
      *v82 = 10;
    }
  }
  v18 = *(_QWORD *)(a1 + 288);
  v19 = *(_DWORD *)(v18 + 256);
  if ( v19 )
  {
    v20 = *(_QWORD *)(v18 + 240);
    v21 = v19 - 1;
    v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v23 = (__int64 *)(v20 + 16LL * v22);
    v24 = *v23;
    if ( a2 == *v23 )
    {
LABEL_25:
      v25 = (__int64 *)v23[1];
      if ( !v25 )
        goto LABEL_58;
      v26 = *(__int64 **)(a1 + 256);
      v27 = (__int64 *)v25[4];
      v28 = *v26;
      if ( a2 != *v27 )
      {
        v29 = *(void (**)())(v28 + 104);
        v30 = (_QWORD *)*v25;
        v31 = 1;
        if ( *v25 )
        {
          do
          {
            v30 = (_QWORD *)*v30;
            ++v31;
          }
          while ( v30 );
        }
        v119.m128i_i32[0] = v31;
        LOWORD(v120) = 265;
        v115.m128i_i64[0] = (__int64)" Depth=";
        LOWORD(v116) = 259;
        v32 = *v27;
        v112 = 266;
        LODWORD(v32) = *(_DWORD *)(v32 + 48);
        v108 = 1;
        v107 = 3;
        LODWORD(v111) = v32;
        v106 = "_";
        LODWORD(v103) = sub_396DD70(a1);
        v104.m128i_i64[0] = (__int64)"  in Loop: Header=BB";
        LOWORD(v105) = 2307;
        v104.m128i_i64[1] = v103;
        v33 = v107;
        if ( v107 )
        {
          if ( v107 == 1 )
          {
            v96 = _mm_loadu_si128(&v104);
            v35 = v112;
            v110 = v105;
            v109 = v96;
            if ( (_BYTE)v112 )
            {
              if ( (_BYTE)v112 != 1 )
              {
                if ( BYTE1(v110) == 1 )
                {
                  v36 = (__m128i *)v109.m128i_i64[0];
                  v37 = 3;
LABEL_36:
                  v38 = v111;
                  if ( HIBYTE(v112) != 1 )
                  {
                    v38 = &v111;
                    v35 = 2;
                  }
                  BYTE1(v114) = v35;
                  v39 = v116;
                  v113.m128i_i64[0] = (__int64)v36;
                  v113.m128i_i64[1] = (__int64)v38;
                  LOBYTE(v114) = v37;
                  if ( (_BYTE)v116 )
                    goto LABEL_39;
                  goto LABEL_94;
                }
LABEL_35:
                v36 = &v109;
                v37 = 2;
                goto LABEL_36;
              }
LABEL_113:
              v37 = v110;
              v113 = _mm_loadu_si128(&v109);
              v114 = v110;
              if ( (_BYTE)v110 )
              {
                v39 = v116;
                if ( (_BYTE)v116 )
                {
                  if ( (_BYTE)v110 == 1 )
                  {
                    v95 = _mm_loadu_si128(&v115);
                    v118 = v116;
                    v117 = v95;
LABEL_117:
                    v37 = v118;
                    if ( (_BYTE)v118 )
                    {
                      v42 = v120;
                      if ( (_BYTE)v120 )
                      {
                        if ( (_BYTE)v118 != 1 )
                        {
LABEL_45:
                          if ( v42 == 1 )
                          {
                            v121 = _mm_loadu_si128(&v117);
                            v122 = v118;
                          }
                          else
                          {
                            v43 = (__m128i *)v117.m128i_i64[0];
                            if ( BYTE1(v118) != 1 )
                            {
                              v43 = &v117;
                              v37 = 2;
                            }
                            v44 = (__m128i *)v119.m128i_i64[0];
                            if ( BYTE1(v120) != 1 )
                            {
                              v44 = &v119;
                              v42 = 2;
                            }
                            v121.m128i_i64[0] = (__int64)v43;
                            v121.m128i_i64[1] = (__int64)v44;
                            LOBYTE(v122) = v37;
                            BYTE1(v122) = v42;
                          }
                          goto LABEL_96;
                        }
                        v121 = _mm_loadu_si128(&v119);
                        v122 = v120;
LABEL_96:
                        if ( v29 != nullsub_580 )
                          ((void (__fastcall *)(__int64 *, __m128i *, __int64))v29)(v26, &v121, 1);
                        goto LABEL_58;
                      }
                    }
LABEL_95:
                    LOWORD(v122) = 256;
                    goto LABEL_96;
                  }
LABEL_39:
                  if ( v39 != 1 )
                  {
                    v40 = (__m128i *)v113.m128i_i64[0];
                    if ( BYTE1(v114) != 1 )
                    {
                      v40 = &v113;
                      v37 = 2;
                    }
                    v41 = (__m128i *)v115.m128i_i64[0];
                    if ( BYTE1(v116) != 1 )
                    {
                      v41 = &v115;
                      v39 = 2;
                    }
                    BYTE1(v118) = v39;
                    v42 = v120;
                    v117.m128i_i64[0] = (__int64)v40;
                    v117.m128i_i64[1] = (__int64)v41;
                    LOBYTE(v118) = v37;
                    if ( (_BYTE)v120 )
                      goto LABEL_45;
                    goto LABEL_95;
                  }
                  v117 = _mm_loadu_si128(&v113);
                  v118 = v114;
                  goto LABEL_117;
                }
              }
LABEL_94:
              LOWORD(v118) = 256;
              goto LABEL_95;
            }
          }
          else
          {
            v34 = (char **)v106;
            if ( v108 != 1 )
            {
              v34 = &v106;
              v33 = 2;
            }
            v109.m128i_i64[1] = (__int64)v34;
            v109.m128i_i64[0] = (__int64)&v104;
            v35 = v112;
            LOBYTE(v110) = 2;
            BYTE1(v110) = v33;
            if ( (_BYTE)v112 )
            {
              if ( (_BYTE)v112 != 1 )
                goto LABEL_35;
              goto LABEL_113;
            }
          }
        }
        else
        {
          LOWORD(v110) = 256;
        }
        LOWORD(v114) = 256;
        goto LABEL_94;
      }
      v83 = (*(__int64 (__fastcall **)(_QWORD))(v28 + 112))(*(_QWORD *)(a1 + 256));
      v84 = sub_396DD70(a1);
      sub_396D290(v83, *v25, v84);
      v85 = *(_WORD **)(v83 + 24);
      if ( *(_QWORD *)(v83 + 16) - (_QWORD)v85 <= 1u )
      {
        sub_16E7EE0(v83, "=>", 2u);
      }
      else
      {
        *v85 = 15933;
        *(_QWORD *)(v83 + 24) += 2LL;
      }
      v86 = (_QWORD *)*v25;
      if ( *v25 )
      {
        v87 = 1;
        do
        {
          v86 = (_QWORD *)*v86;
          v88 = v87++;
        }
        while ( v86 );
        v89 = 2 * v88;
      }
      else
      {
        v89 = 0;
      }
      sub_16E8750(v83, v89);
      v90 = *(_QWORD *)(v83 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v83 + 16) - v90) <= 4 )
      {
        sub_16E7EE0(v83, "This ", 5u);
      }
      else
      {
        *(_DWORD *)v90 = 1936287828;
        *(_BYTE *)(v90 + 4) = 32;
        *(_QWORD *)(v83 + 24) += 5LL;
      }
      if ( v25[2] == v25[1] )
      {
        v99 = *(_QWORD *)(v83 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v83 + 16) - v99) <= 5 )
        {
          sub_16E7EE0(v83, "Inner ", 6u);
        }
        else
        {
          *(_DWORD *)v99 = 1701736009;
          *(_WORD *)(v99 + 4) = 8306;
          *(_QWORD *)(v83 + 24) += 6LL;
        }
      }
      v45 = (_QWORD *)*v25;
      v46 = 1;
      if ( *v25 )
      {
        do
        {
          v45 = (_QWORD *)*v45;
          ++v46;
        }
        while ( v45 );
      }
      v119.m128i_i32[0] = v46;
      v121.m128i_i64[0] = (__int64)"Loop Header: Depth=";
      LOWORD(v122) = 2307;
      v121.m128i_i64[1] = v119.m128i_i64[0];
      sub_16E2CE0((__int64)&v121, v83);
      v47 = *(_BYTE **)(v83 + 24);
      if ( (unsigned __int64)v47 >= *(_QWORD *)(v83 + 16) )
      {
        sub_16E7DE0(v83, 10);
      }
      else
      {
        *(_QWORD *)(v83 + 24) = v47 + 1;
        *v47 = 10;
      }
      v48 = sub_396DD70(a1);
      sub_396C890(v83, (__int64)v25, v48);
    }
    else
    {
      v97 = 1;
      while ( v24 != -8 )
      {
        v98 = v97 + 1;
        v22 = v21 & (v97 + v22);
        v23 = (__int64 *)(v20 + 16LL * v22);
        v24 = *v23;
        if ( a2 == *v23 )
          goto LABEL_25;
        v97 = v98;
      }
    }
  }
LABEL_58:
  if ( *(_QWORD *)(a2 + 72) == *(_QWORD *)(a2 + 64) )
  {
LABEL_61:
    if ( *(_BYTE *)(a1 + 416) )
    {
      v72 = *(_QWORD *)(a1 + 256);
      v73 = *(void (__fastcall **)(__int64, __m128i *, _QWORD))(*(_QWORD *)v72 + 120LL);
      v117.m128i_i32[0] = *(_DWORD *)(a2 + 48);
      v119.m128i_i64[0] = (__int64)" %bb.";
      v121.m128i_i64[1] = (__int64)":";
      v119.m128i_i64[1] = v117.m128i_i64[0];
      LOWORD(v120) = 2563;
      v121.m128i_i64[0] = (__int64)&v119;
      LOWORD(v122) = 770;
      v73(v72, &v121, 0);
    }
    return;
  }
LABEL_59:
  if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 320LL))(a1, a2) && !*(_BYTE *)(a2 + 183) )
    goto LABEL_61;
  v49 = *(_QWORD *)(a1 + 256);
  v50 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v49 + 176LL);
  v51 = sub_1DD5A70(a2);
  v50(v49, v51, 0);
  v52 = *(_QWORD *)(a1 + 528);
  v53 = *(_DWORD *)(v52 + 256);
  if ( !v53 )
    return;
  v54 = *(_QWORD *)(v52 + 240);
  v55 = v53 - 1;
  v56 = (v53 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v57 = (__int64 *)(v54 + 16LL * v56);
  v58 = *v57;
  if ( a2 == *v57 )
  {
LABEL_67:
    v59 = v57[1];
    if ( !v59 )
      return;
    if ( a2 != **(_QWORD **)(v59 + 32) )
      return;
    v60 = sub_1E29AC0(v59);
    if ( !v60 )
      return;
    v61 = *(_QWORD *)(v60 + 40);
    if ( !v61 )
      return;
    v62 = sub_157EBA0(v61);
    v63 = v62;
    if ( *(_BYTE *)(v62 + 16) != 26 || (*(_DWORD *)(v62 + 20) & 0xFFFFFFF) == 1 )
      return;
    if ( !*(_QWORD *)(v62 + 48) && *(__int16 *)(v62 + 18) >= 0 )
      goto LABEL_141;
    v64 = sub_1625940(v62, "pragma", 6u);
    if ( v64 )
    {
      if ( *(_DWORD *)(v64 + 8) == 2 )
      {
        v91 = *(_BYTE **)(v64 - 16);
        if ( !*v91 )
        {
          v92 = sub_161E970((__int64)v91);
          if ( v93 == 6 && *(_DWORD *)v92 == 1869770357 && *(_WORD *)(v92 + 4) == 27756 )
            goto LABEL_112;
        }
      }
    }
    if ( !*(_QWORD *)(v63 + 48) )
    {
LABEL_141:
      if ( *(__int16 *)(v63 + 18) >= 0 )
        return;
    }
    v65 = sub_1625940(v63, "llvm.loop", 9u);
    v66 = v65;
    if ( !v65 )
      return;
    v67 = *(_DWORD *)(v65 + 8);
    if ( v67 <= 1 )
      return;
    v68 = 1;
    v69 = *(_BYTE **)(v65 - 8LL * v67);
    if ( *v69 )
      goto LABEL_83;
LABEL_80:
    v70 = sub_161E970((__int64)v69);
    if ( v71 <= 0x10
      || *(_QWORD *)v70 ^ 0x6F6F6C2E6D766C6CLL | *(_QWORD *)(v70 + 8) ^ 0x6C6C6F726E752E70LL
      || *(_BYTE *)(v70 + 16) != 46 )
    {
LABEL_83:
      while ( v67 != ++v68 )
      {
        v69 = *(_BYTE **)(v66 - 8LL * *(unsigned int *)(v66 + 8));
        if ( !*v69 )
          goto LABEL_80;
      }
      return;
    }
LABEL_112:
    v94 = *(__int64 **)(a1 + 256);
    v119.m128i_i64[1] = 21;
    v119.m128i_i64[0] = (__int64)"\t.pragma \"nounroll\";\n";
    LOWORD(v122) = 261;
    v121.m128i_i64[0] = (__int64)&v119;
    sub_38DD5A0(v94, (__int64)&v121);
    return;
  }
  v100 = 1;
  while ( v58 != -8 )
  {
    v101 = v100 + 1;
    v56 = v55 & (v100 + v56);
    v57 = (__int64 *)(v54 + 16LL * v56);
    v58 = *v57;
    if ( a2 == *v57 )
      goto LABEL_67;
    v100 = v101;
  }
}

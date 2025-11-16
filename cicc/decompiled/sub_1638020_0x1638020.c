// Function: sub_1638020
// Address: 0x1638020
//
__int64 __fastcall sub_1638020(__int64 a1, void *a2, __int64 a3)
{
  __int64 v3; // r12
  char *v5; // rax
  unsigned int v6; // esi
  __int64 v7; // rdi
  unsigned int v8; // ecx
  _QWORD *v9; // rax
  void *v10; // rdx
  _QWORD *v11; // r13
  _QWORD *v12; // rdx
  int v13; // esi
  unsigned int v14; // r9d
  __int64 *v15; // rax
  __int64 v16; // r10
  __int64 v17; // rsi
  char v18; // al
  __int64 v19; // r12
  char v20; // cl
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 result; // rax
  _QWORD *v24; // r15
  _QWORD *v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // r14
  __int64 v28; // rbx
  __int64 v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // rdi
  _QWORD *v32; // r13
  __int64 v33; // rdi
  int v34; // edx
  int v35; // edx
  int v36; // r8d
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rsi
  unsigned int j; // eax
  _QWORD *v41; // rsi
  unsigned int v42; // eax
  int v43; // eax
  __int64 v44; // rax
  __m128i *v45; // rdx
  __int64 v46; // r13
  __m128i v47; // xmm0
  __int64 v48; // rdx
  __int64 v49; // rcx
  unsigned int v50; // esi
  __int64 *v51; // rax
  __int64 v52; // r10
  __int64 v53; // rax
  size_t v54; // rdx
  _DWORD *v55; // rdi
  const char *v56; // rsi
  unsigned __int64 v57; // rax
  __int64 v58; // rax
  size_t v59; // rdx
  _BYTE *v60; // rdi
  const char *v61; // rsi
  _BYTE *v62; // rax
  int v63; // eax
  __int64 v64; // rax
  __m128i *v65; // rdx
  __int64 v66; // r13
  __m128i si128; // xmm0
  __int64 v68; // rax
  size_t v69; // rdx
  _BYTE *v70; // rdi
  const char *v71; // rsi
  _BYTE *v72; // rax
  size_t v73; // r14
  int v74; // eax
  int v75; // r8d
  __int64 v76; // rax
  __int64 v77; // rax
  int v78; // r10d
  _QWORD *v79; // rax
  int v80; // ebx
  int v81; // ecx
  int v82; // edx
  __int64 v83; // rsi
  int v84; // edi
  void **v85; // r13
  void *v86; // rcx
  void **i; // rbx
  void **v88; // r15
  void *v89; // rdi
  int v90; // r8d
  int v91; // r8d
  int v92; // eax
  int v93; // esi
  __int64 v94; // r8
  unsigned int v95; // edx
  __int64 v96; // rdi
  int v97; // r10d
  _QWORD *v98; // r9
  int v99; // eax
  int v100; // edx
  __int64 v101; // rdi
  _QWORD *v102; // r8
  unsigned int v103; // r13d
  int v104; // r9d
  __int64 v105; // rsi
  _QWORD *v106; // r11
  size_t v107; // [rsp+0h] [rbp-140h]
  size_t v108; // [rsp+0h] [rbp-140h]
  __int64 v109; // [rsp+8h] [rbp-138h]
  _QWORD *v110; // [rsp+10h] [rbp-130h]
  _QWORD *v111; // [rsp+20h] [rbp-120h]
  void *v112; // [rsp+28h] [rbp-118h]
  _QWORD v113[2]; // [rsp+30h] [rbp-110h] BYREF
  __int64 v114; // [rsp+40h] [rbp-100h] BYREF
  char v115[8]; // [rsp+48h] [rbp-F8h] BYREF
  char v116[48]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v117; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v118; // [rsp+88h] [rbp-B8h]
  _QWORD *v119; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v120; // [rsp+98h] [rbp-A8h]
  char v121; // [rsp+110h] [rbp-30h] BYREF

  v3 = a1;
  v112 = a2;
  if ( *(_DWORD *)(a3 + 88) != *(_DWORD *)(a3 + 84)
    || (result = sub_134EB50(a3, (__int64)&unk_4F9EE48), !(_DWORD)result)
    && (a2 = &unk_4F9EE68, a1 = a3, result = sub_134EB50(a3, (__int64)&unk_4F9EE68), !(_DWORD)result) )
  {
    if ( *(_BYTE *)(v3 + 96) )
    {
      v64 = sub_16BA580(a1, a2, a3);
      v65 = *(__m128i **)(v64 + 24);
      v66 = v64;
      if ( *(_QWORD *)(v64 + 16) - (_QWORD)v65 <= 0x2Cu )
      {
        v66 = sub_16E7EE0(v64, "Invalidating all non-preserved analyses for: ", 45);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_42ABC30);
        qmemcpy(&v65[2], "nalyses for: ", 13);
        *v65 = si128;
        v65[1] = _mm_load_si128(xmmword_42ABC40);
        *(_QWORD *)(v64 + 24) += 45LL;
      }
      v68 = sub_1649960(v112);
      v70 = *(_BYTE **)(v66 + 24);
      v71 = (const char *)v68;
      v72 = *(_BYTE **)(v66 + 16);
      v73 = v69;
      if ( v72 - v70 < v69 )
      {
        v66 = sub_16E7EE0(v66, v71, v69);
        v72 = *(_BYTE **)(v66 + 16);
        v70 = *(_BYTE **)(v66 + 24);
      }
      else if ( v69 )
      {
        memcpy(v70, v71, v69);
        v72 = *(_BYTE **)(v66 + 16);
        v70 = (_BYTE *)(v73 + *(_QWORD *)(v66 + 24));
        *(_QWORD *)(v66 + 24) = v70;
      }
      if ( v72 == v70 )
      {
        sub_16E7EE0(v66, "\n", 1);
      }
      else
      {
        *v70 = 10;
        ++*(_QWORD *)(v66 + 24);
      }
    }
    v5 = (char *)&v119;
    v117 = 0;
    v118 = 1;
    do
    {
      *(_QWORD *)v5 = -8;
      v5 += 16;
    }
    while ( v5 != &v121 );
    v6 = *(_DWORD *)(v3 + 56);
    v113[0] = &v117;
    v113[1] = v3 + 64;
    if ( v6 )
    {
      v7 = *(_QWORD *)(v3 + 40);
      v8 = (v6 - 1) & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
      v9 = (_QWORD *)(v7 + 32LL * v8);
      v10 = (void *)*v9;
      v110 = v9;
      if ( v112 == (void *)*v9 )
      {
        v11 = (_QWORD *)v9[1];
        v111 = v9 + 1;
        goto LABEL_8;
      }
      v78 = 1;
      v79 = 0;
      while ( v10 != (void *)-8LL )
      {
        if ( v10 != (void *)-16LL || v79 )
          v110 = v79;
        v8 = (v6 - 1) & (v78 + v8);
        v106 = (_QWORD *)(v7 + 32LL * v8);
        v10 = (void *)*v106;
        if ( v112 == (void *)*v106 )
        {
          v110 = (_QWORD *)(v7 + 32LL * v8);
          v11 = (_QWORD *)v106[1];
          v111 = v106 + 1;
LABEL_8:
          if ( v111 == v11 )
          {
LABEL_100:
            result = (unsigned int)v118 >> 1;
            if ( !((unsigned int)v118 >> 1) )
            {
LABEL_43:
              if ( v111 != v11 )
              {
LABEL_44:
                if ( (v118 & 1) == 0 )
                  return j___libc_free_0(v119);
                return result;
              }
            }
LABEL_101:
            result = *(unsigned int *)(v3 + 56);
            if ( (_DWORD)result )
            {
              v82 = result - 1;
              v83 = *(_QWORD *)(v3 + 40);
              v84 = 1;
              result = ((_DWORD)result - 1) & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
              v85 = (void **)(v83 + 32 * result);
              v86 = *v85;
              if ( v112 == *v85 )
              {
LABEL_103:
                for ( i = (void **)v85[1]; v85 + 1 != i; result = j_j___libc_free_0(v88, 32) )
                {
                  v88 = i;
                  i = (void **)*i;
                  v89 = v88[3];
                  if ( v89 )
                    (*(void (__fastcall **)(void *))(*(_QWORD *)v89 + 8LL))(v89);
                }
                *v85 = (void *)-16LL;
                --*(_DWORD *)(v3 + 48);
                ++*(_DWORD *)(v3 + 52);
              }
              else
              {
                while ( v86 != (void *)-8LL )
                {
                  result = v82 & (unsigned int)(v84 + result);
                  v85 = (void **)(v83 + 32LL * (unsigned int)result);
                  v86 = *v85;
                  if ( v112 == *v85 )
                    goto LABEL_103;
                  ++v84;
                }
              }
            }
            goto LABEL_44;
          }
          v109 = v3;
          while ( 2 )
          {
            v19 = v11[2];
            v20 = v118 & 1;
            if ( (v118 & 1) != 0 )
            {
              v12 = &v119;
              v13 = 7;
              goto LABEL_11;
            }
            v21 = v120;
            v12 = v119;
            if ( v120 )
            {
              v13 = v120 - 1;
LABEL_11:
              v14 = v13 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
              v15 = &v12[2 * v14];
              v16 = *v15;
              if ( v19 == *v15 )
              {
LABEL_12:
                v17 = 16;
                if ( !v20 )
                  v17 = 2LL * v120;
                if ( v15 == &v12[v17] )
                {
                  v18 = (*(__int64 (__fastcall **)(_QWORD, void *, __int64, _QWORD *))(*(_QWORD *)v11[3] + 16LL))(
                          v11[3],
                          v112,
                          a3,
                          v113);
                  v114 = v19;
                  v115[0] = v18;
                  sub_1367360((__int64)v116, (__int64)&v117, &v114, v115);
                }
                v11 = (_QWORD *)*v11;
                if ( v11 == v111 )
                {
                  v3 = v109;
                  v11 = (_QWORD *)v110[1];
                  result = (unsigned int)v118 >> 1;
                  if ( !((unsigned int)v118 >> 1) )
                    goto LABEL_43;
                  if ( v11 == v111 )
                    goto LABEL_101;
                  v24 = (_QWORD *)v110[1];
                  while ( 2 )
                  {
                    if ( (v118 & 1) != 0 )
                    {
                      v25 = &v119;
                      v26 = 7;
                      goto LABEL_28;
                    }
                    v25 = v119;
                    v26 = v120 - 1;
                    if ( !v120 )
                    {
LABEL_52:
                      v24 = (_QWORD *)*v24;
                      goto LABEL_41;
                    }
LABEL_28:
                    v27 = v24[2];
                    v28 = ((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4);
                    v29 = (unsigned int)v26 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
                    v30 = &v25[2 * v29];
                    v31 = *v30;
                    if ( v27 != *v30 )
                    {
                      v74 = 1;
                      while ( v31 != -8 )
                      {
                        v75 = v74 + 1;
                        v29 = (unsigned int)v26 & (v74 + (_DWORD)v29);
                        v30 = &v25[2 * (unsigned int)v29];
                        v31 = *v30;
                        if ( v27 == *v30 )
                          goto LABEL_29;
                        v74 = v75;
                      }
                      goto LABEL_52;
                    }
LABEL_29:
                    if ( !*((_BYTE *)v30 + 8) )
                      goto LABEL_52;
                    if ( !*(_BYTE *)(v109 + 96) )
                    {
LABEL_31:
                      v32 = (_QWORD *)*v24;
                      --v110[3];
                      sub_2208CA0(v24);
                      v33 = v24[3];
                      if ( v33 )
                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v33 + 8LL))(v33);
                      j_j___libc_free_0(v24, 32);
                      v34 = *(_DWORD *)(v109 + 88);
                      if ( v34 )
                      {
                        v35 = v34 - 1;
                        v36 = 1;
                        v37 = (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4) | (unsigned __int64)(v28 << 32))
                            - 1
                            - ((unsigned __int64)(((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4)) << 32);
                        v38 = ((v37 >> 22) ^ v37) - 1 - (((v37 >> 22) ^ v37) << 13);
                        v39 = ((9 * ((v38 >> 8) ^ v38)) >> 15) ^ (9 * ((v38 >> 8) ^ v38));
                        for ( j = v35 & (((v39 - 1 - (v39 << 27)) >> 31) ^ (v39 - 1 - ((_DWORD)v39 << 27))); ; j = v35 & v42 )
                        {
                          v41 = (_QWORD *)(*(_QWORD *)(v109 + 72) + 24LL * j);
                          if ( v27 == *v41 && v112 == (void *)v41[1] )
                            break;
                          if ( *v41 == -8 && v41[1] == -8 )
                            goto LABEL_40;
                          v42 = v36 + j;
                          ++v36;
                        }
                        *v41 = -16;
                        v41[1] = -16;
                        --*(_DWORD *)(v109 + 80);
                        ++*(_DWORD *)(v109 + 84);
                      }
LABEL_40:
                      v24 = v32;
LABEL_41:
                      if ( v24 == v111 )
                      {
                        result = (__int64)v110;
                        v11 = (_QWORD *)v110[1];
                        goto LABEL_43;
                      }
                      continue;
                    }
                    break;
                  }
                  v44 = sub_16BA580(v31, v29, v26);
                  v45 = *(__m128i **)(v44 + 24);
                  v46 = v44;
                  if ( *(_QWORD *)(v44 + 16) - (_QWORD)v45 <= 0x16u )
                  {
                    v46 = sub_16E7EE0(v44, "Invalidating analysis: ", 23);
                  }
                  else
                  {
                    v47 = _mm_load_si128((const __m128i *)&xmmword_42ABC20);
                    v45[1].m128i_i32[0] = 1769175404;
                    v45[1].m128i_i16[2] = 14963;
                    v45[1].m128i_i8[6] = 32;
                    *v45 = v47;
                    *(_QWORD *)(v44 + 24) += 23LL;
                  }
                  v48 = *(unsigned int *)(v109 + 24);
                  v49 = *(_QWORD *)(v109 + 8);
                  if ( (_DWORD)v48 )
                  {
                    v50 = (v48 - 1) & v28;
                    v51 = (__int64 *)(v49 + 16LL * v50);
                    v52 = *v51;
                    if ( v27 == *v51 )
                    {
LABEL_57:
                      v53 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v51[1] + 24LL))(v51[1]);
                      v55 = *(_DWORD **)(v46 + 24);
                      v56 = (const char *)v53;
                      v57 = *(_QWORD *)(v46 + 16) - (_QWORD)v55;
                      if ( v54 > v57 )
                      {
                        v76 = sub_16E7EE0(v46, v56);
                        v55 = *(_DWORD **)(v76 + 24);
                        v46 = v76;
                        v57 = *(_QWORD *)(v76 + 16) - (_QWORD)v55;
                      }
                      else if ( v54 )
                      {
                        v108 = v54;
                        memcpy(v55, v56, v54);
                        v77 = *(_QWORD *)(v46 + 16);
                        v55 = (_DWORD *)(v108 + *(_QWORD *)(v46 + 24));
                        *(_QWORD *)(v46 + 24) = v55;
                        v57 = v77 - (_QWORD)v55;
                      }
                      if ( v57 <= 3 )
                      {
                        v46 = sub_16E7EE0(v46, " on ", 4);
                      }
                      else
                      {
                        *v55 = 544108320;
                        *(_QWORD *)(v46 + 24) += 4LL;
                      }
                      v58 = sub_1649960(v112);
                      v60 = *(_BYTE **)(v46 + 24);
                      v61 = (const char *)v58;
                      v62 = *(_BYTE **)(v46 + 16);
                      if ( v59 > v62 - v60 )
                      {
                        v46 = sub_16E7EE0(v46, v61);
                        v62 = *(_BYTE **)(v46 + 16);
                        v60 = *(_BYTE **)(v46 + 24);
                      }
                      else if ( v59 )
                      {
                        v107 = v59;
                        memcpy(v60, v61, v59);
                        v62 = *(_BYTE **)(v46 + 16);
                        v60 = (_BYTE *)(v107 + *(_QWORD *)(v46 + 24));
                        *(_QWORD *)(v46 + 24) = v60;
                      }
                      if ( v62 == v60 )
                      {
                        sub_16E7EE0(v46, "\n", 1);
                      }
                      else
                      {
                        *v60 = 10;
                        ++*(_QWORD *)(v46 + 24);
                      }
                      goto LABEL_31;
                    }
                    v63 = 1;
                    while ( v52 != -8 )
                    {
                      v91 = v63 + 1;
                      v50 = (v48 - 1) & (v63 + v50);
                      v51 = (__int64 *)(v49 + 16LL * v50);
                      v52 = *v51;
                      if ( v27 == *v51 )
                        goto LABEL_57;
                      v63 = v91;
                    }
                  }
                  v51 = (__int64 *)(v49 + 16 * v48);
                  goto LABEL_57;
                }
                continue;
              }
              v43 = 1;
              while ( v16 != -8 )
              {
                v90 = v43 + 1;
                v14 = v13 & (v43 + v14);
                v15 = &v12[2 * v14];
                v16 = *v15;
                if ( v19 == *v15 )
                  goto LABEL_12;
                v43 = v90;
              }
              if ( v20 )
              {
                v22 = 16;
                goto LABEL_22;
              }
              v21 = v120;
            }
            break;
          }
          v22 = 2 * v21;
LABEL_22:
          v15 = &v12[v22];
          goto LABEL_12;
        }
        ++v78;
        v79 = v110;
        v110 = (_QWORD *)(v7 + 32LL * v8);
      }
      v80 = *(_DWORD *)(v3 + 48);
      if ( !v79 )
        v79 = v110;
      ++*(_QWORD *)(v3 + 32);
      v81 = v80 + 1;
      if ( 4 * (v80 + 1) < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(v3 + 52) - v81 > v6 >> 3 )
        {
LABEL_97:
          *(_DWORD *)(v3 + 48) = v81;
          if ( *v79 != -8 )
            --*(_DWORD *)(v3 + 52);
          v11 = v79 + 1;
          v79[3] = 0;
          v79[2] = v79 + 1;
          *v79 = v112;
          v79[1] = v79 + 1;
          v111 = v79 + 1;
          goto LABEL_100;
        }
        sub_1637DA0(v3 + 32, v6);
        v99 = *(_DWORD *)(v3 + 56);
        if ( v99 )
        {
          v100 = v99 - 1;
          v101 = *(_QWORD *)(v3 + 40);
          v102 = 0;
          v103 = (v99 - 1) & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
          v104 = 1;
          v81 = *(_DWORD *)(v3 + 48) + 1;
          v79 = (_QWORD *)(v101 + 32LL * v103);
          v105 = *v79;
          if ( v112 != (void *)*v79 )
          {
            while ( v105 != -8 )
            {
              if ( !v102 && v105 == -16 )
                v102 = v79;
              v103 = v100 & (v104 + v103);
              v79 = (_QWORD *)(v101 + 32LL * v103);
              v105 = *v79;
              if ( v112 == (void *)*v79 )
                goto LABEL_97;
              ++v104;
            }
            if ( v102 )
              v79 = v102;
          }
          goto LABEL_97;
        }
LABEL_148:
        ++*(_DWORD *)(v3 + 48);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(v3 + 32);
    }
    sub_1637DA0(v3 + 32, 2 * v6);
    v92 = *(_DWORD *)(v3 + 56);
    if ( v92 )
    {
      v93 = v92 - 1;
      v94 = *(_QWORD *)(v3 + 40);
      v81 = *(_DWORD *)(v3 + 48) + 1;
      v95 = (v92 - 1) & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
      v79 = (_QWORD *)(v94 + 32LL * v95);
      v96 = *v79;
      if ( v112 != (void *)*v79 )
      {
        v97 = 1;
        v98 = 0;
        while ( v96 != -8 )
        {
          if ( !v98 && v96 == -16 )
            v98 = v79;
          v95 = v93 & (v97 + v95);
          v79 = (_QWORD *)(v94 + 32LL * v95);
          v96 = *v79;
          if ( v112 == (void *)*v79 )
            goto LABEL_97;
          ++v97;
        }
        if ( v98 )
          v79 = v98;
      }
      goto LABEL_97;
    }
    goto LABEL_148;
  }
  return result;
}

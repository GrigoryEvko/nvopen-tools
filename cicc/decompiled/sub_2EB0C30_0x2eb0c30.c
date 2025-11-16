// Function: sub_2EB0C30
// Address: 0x2eb0c30
//
__int64 __fastcall sub_2EB0C30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rax
  unsigned int v5; // esi
  __int64 v6; // r9
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 v9; // rdx
  unsigned __int64 v10; // r15
  __int64 *v11; // r8
  int v12; // esi
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rbx
  char v18; // cl
  __int64 v19; // rsi
  char v20; // r8
  __int64 *v21; // rdi
  int v22; // esi
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // r10
  int v26; // r11d
  __int64 *v27; // r9
  unsigned int v28; // eax
  unsigned int v29; // edx
  unsigned int v30; // esi
  __int64 *v31; // rsi
  int v32; // ecx
  unsigned int v33; // edx
  __int64 v34; // rdi
  int v35; // r11d
  __int64 *v36; // r10
  __int64 v37; // rsi
  __int64 result; // rax
  __int64 v39; // r13
  __int64 *v40; // rsi
  int v41; // edx
  unsigned int v42; // edi
  __int64 *v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rcx
  __int64 v46; // rax
  int v47; // r9d
  unsigned int i; // edi
  __int64 v49; // rdx
  unsigned int v50; // edi
  int v51; // eax
  int v52; // eax
  __int64 v53; // rcx
  __int64 v54; // rdx
  __int64 v55; // rsi
  unsigned int v56; // edi
  __int64 *v57; // rax
  __int64 v58; // r10
  __int64 v59; // r12
  __int64 v60; // rax
  _QWORD *v61; // rbx
  _QWORD *v62; // rax
  _QWORD *v63; // r13
  _QWORD *v64; // rax
  _QWORD *v65; // rdi
  _QWORD *v66; // rdi
  __int64 v67; // rdx
  __int64 v68; // rsi
  unsigned __int64 v69; // rbx
  __int64 v70; // rdi
  int v71; // edx
  int v72; // r8d
  int v73; // esi
  unsigned int j; // eax
  _QWORD *v75; // rdx
  unsigned int v76; // eax
  int v77; // eax
  void **v78; // rdx
  __int64 *v79; // rcx
  int v80; // r11d
  __int64 *v81; // rax
  int v82; // ecx
  __int64 v83; // rcx
  int v84; // edx
  __int64 *v85; // r12
  __int64 v86; // rsi
  __int64 *v87; // rbx
  unsigned __int64 v88; // r14
  __int64 v89; // rdi
  __int64 *v90; // rsi
  int v91; // ecx
  int v92; // r11d
  unsigned int v93; // edx
  __int64 v94; // rdi
  int v95; // r10d
  int v96; // r8d
  int v97; // r9d
  int v98; // eax
  int v99; // esi
  __int64 v100; // r8
  unsigned int v101; // edx
  __int64 v102; // rdi
  int v103; // r10d
  __int64 *v104; // r9
  int v105; // eax
  int v106; // edx
  __int64 v107; // rdi
  int v108; // r9d
  unsigned int v109; // ebx
  __int64 *v110; // r8
  __int64 v111; // rsi
  int v112; // edi
  __int64 v113; // [rsp+8h] [rbp-128h]
  char v114; // [rsp+18h] [rbp-118h]
  __int64 v115; // [rsp+18h] [rbp-118h]
  char v116; // [rsp+18h] [rbp-118h]
  __int64 *v117; // [rsp+20h] [rbp-110h]
  __int64 *v119; // [rsp+30h] [rbp-100h]
  _QWORD *v120; // [rsp+38h] [rbp-F8h]
  _QWORD *v122; // [rsp+58h] [rbp-D8h] BYREF
  _QWORD v123[2]; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v124; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v125; // [rsp+78h] [rbp-B8h]
  __int64 *v126; // [rsp+80h] [rbp-B0h] BYREF
  unsigned int v127; // [rsp+88h] [rbp-A8h]
  char v128; // [rsp+100h] [rbp-30h] BYREF

  if ( *(_DWORD *)(a3 + 72) == *(_DWORD *)(a3 + 68) )
  {
    if ( *(_BYTE *)(a3 + 28) )
    {
      v78 = *(void ***)(a3 + 8);
      v79 = (__int64 *)&v78[*(unsigned int *)(a3 + 20)];
      if ( v78 != (void **)v79 )
      {
        result = (__int64)v78;
        while ( *(__int64 **)result != &qword_4F82400 )
        {
          result += 8;
          if ( v79 == (__int64 *)result )
            goto LABEL_164;
        }
        return result;
      }
    }
    else
    {
      result = (__int64)sub_C8CA60(a3, (__int64)&qword_4F82400);
      if ( result )
        return result;
      if ( *(_BYTE *)(a3 + 28) )
      {
        v78 = *(void ***)(a3 + 8);
        result = (__int64)&v78[*(unsigned int *)(a3 + 20)];
        if ( v78 != (void **)result )
        {
LABEL_164:
          while ( *v78 != &unk_4FDC268 )
          {
            if ( ++v78 == (void **)result )
              goto LABEL_2;
          }
          return result;
        }
      }
      else
      {
        result = (__int64)sub_C8CA60(a3, (__int64)&unk_4FDC268);
        if ( result )
          return result;
      }
    }
  }
LABEL_2:
  v4 = (__int64 *)&v126;
  v124 = 0;
  v125 = 1;
  do
  {
    *v4 = -4096;
    v4 += 2;
  }
  while ( v4 != (__int64 *)&v128 );
  v123[0] = &v124;
  v5 = *(_DWORD *)(a1 + 56);
  v6 = a1 + 32;
  v123[1] = a1 + 64;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_154;
  }
  v7 = *(_QWORD *)(a1 + 40);
  v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = *(_QWORD *)(v7 + 32LL * v8);
  v117 = (__int64 *)(v7 + 32LL * v8);
  if ( a2 == v9 )
  {
LABEL_6:
    v10 = v117[1];
    v120 = v117 + 1;
    if ( v117 + 1 != (__int64 *)v10 )
    {
      v119 = &v124;
      while ( 1 )
      {
        v17 = *(_QWORD *)(v10 + 16);
        v18 = v125 & 1;
        if ( (v125 & 1) != 0 )
        {
          v11 = (__int64 *)&v126;
          v12 = 7;
        }
        else
        {
          v19 = v127;
          v11 = v126;
          if ( !v127 )
            goto LABEL_37;
          v12 = v127 - 1;
        }
        v13 = v12 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v14 = &v11[2 * v13];
        v15 = *v14;
        if ( v17 == *v14 )
          goto LABEL_10;
        v52 = 1;
        while ( v15 != -4096 )
        {
          v95 = v52 + 1;
          v13 = v12 & (v52 + v13);
          v14 = &v11[2 * v13];
          v15 = *v14;
          if ( v17 == *v14 )
            goto LABEL_10;
          v52 = v95;
        }
        if ( v18 )
        {
          v37 = 16;
          goto LABEL_38;
        }
        v19 = v127;
LABEL_37:
        v37 = 2 * v19;
LABEL_38:
        v14 = &v11[v37];
LABEL_10:
        v16 = 16;
        if ( !v18 )
          v16 = 2LL * v127;
        if ( v14 != &v11[v16] )
          goto LABEL_13;
        v20 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD *))(**(_QWORD **)(v10 + 24) + 16LL))(
                *(_QWORD *)(v10 + 24),
                a2,
                a3,
                v123);
        if ( (v125 & 1) != 0 )
        {
          v21 = (__int64 *)&v126;
          v22 = 7;
        }
        else
        {
          v30 = v127;
          v21 = v126;
          if ( !v127 )
          {
            v28 = v125;
            ++v124;
            v27 = 0;
            v29 = ((unsigned int)v125 >> 1) + 1;
            goto LABEL_60;
          }
          v22 = v127 - 1;
        }
        v23 = v22 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v24 = &v21[2 * v23];
        v25 = *v24;
        if ( v17 != *v24 )
        {
          v26 = 1;
          v27 = 0;
          while ( v25 != -4096 )
          {
            if ( v27 || v25 != -8192 )
              v24 = v27;
            v23 = v22 & (v26 + v23);
            v25 = v21[2 * v23];
            if ( v17 == v25 )
              goto LABEL_13;
            ++v26;
            v27 = v24;
            v24 = &v21[2 * v23];
          }
          if ( !v27 )
            v27 = v24;
          v28 = v125;
          ++v124;
          v29 = ((unsigned int)v125 >> 1) + 1;
          if ( (v125 & 1) != 0 )
          {
            v30 = 8;
            if ( 4 * v29 >= 0x18 )
              goto LABEL_26;
LABEL_61:
            if ( v30 - HIDWORD(v125) - v29 <= v30 >> 3 )
            {
              v116 = v20;
              sub_BBCB10((__int64)&v124, v30);
              v20 = v116;
              if ( (v125 & 1) != 0 )
              {
                v90 = (__int64 *)&v126;
                v91 = 7;
              }
              else
              {
                v90 = v126;
                if ( !v127 )
                {
LABEL_202:
                  LODWORD(v125) = (2 * ((unsigned int)v125 >> 1) + 2) | v125 & 1;
                  BUG();
                }
                v91 = v127 - 1;
              }
              v92 = 1;
              v36 = 0;
              v28 = v125;
              v93 = v91 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
              v27 = &v90[2 * v93];
              v94 = *v27;
              if ( v17 != *v27 )
              {
                while ( v94 != -4096 )
                {
                  if ( !v36 && v94 == -8192 )
                    v36 = v27;
                  v93 = v91 & (v92 + v93);
                  v27 = &v90[2 * v93];
                  v94 = *v27;
                  if ( v17 == *v27 )
                    goto LABEL_33;
                  ++v92;
                }
LABEL_31:
                if ( v36 )
                  v27 = v36;
LABEL_33:
                v28 = v125;
              }
            }
          }
          else
          {
            v30 = v127;
LABEL_60:
            if ( 4 * v29 < 3 * v30 )
              goto LABEL_61;
LABEL_26:
            v114 = v20;
            sub_BBCB10((__int64)&v124, 2 * v30);
            v20 = v114;
            if ( (v125 & 1) != 0 )
            {
              v31 = (__int64 *)&v126;
              v32 = 7;
            }
            else
            {
              v31 = v126;
              if ( !v127 )
                goto LABEL_202;
              v32 = v127 - 1;
            }
            v28 = v125;
            v33 = v32 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v27 = &v31[2 * v33];
            v34 = *v27;
            if ( v17 != *v27 )
            {
              v35 = 1;
              v36 = 0;
              while ( v34 != -4096 )
              {
                if ( !v36 && v34 == -8192 )
                  v36 = v27;
                v33 = v32 & (v35 + v33);
                v27 = &v31[2 * v33];
                v34 = *v27;
                if ( v17 == *v27 )
                  goto LABEL_33;
                ++v35;
              }
              goto LABEL_31;
            }
          }
          LODWORD(v125) = (2 * (v28 >> 1) + 2) | v28 & 1;
          if ( *v27 != -4096 )
            --HIDWORD(v125);
          *v27 = v17;
          *((_BYTE *)v27 + 8) = v20;
        }
LABEL_13:
        v10 = *(_QWORD *)v10;
        if ( (_QWORD *)v10 == v120 )
        {
          v10 = v117[1];
          result = (unsigned int)v125 >> 1;
          if ( !((unsigned int)v125 >> 1) )
            goto LABEL_56;
          if ( v120 != (_QWORD *)v10 )
          {
            v113 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
            while ( 1 )
            {
              v39 = *(_QWORD *)(v10 + 16);
              if ( (v125 & 1) != 0 )
              {
                v40 = (__int64 *)&v126;
                v41 = 7;
              }
              else
              {
                v40 = v126;
                if ( !v127 )
                  goto LABEL_53;
                v41 = v127 - 1;
              }
              LODWORD(v119) = ((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4);
              v42 = v41 & (unsigned int)v119;
              v43 = &v40[2 * (v41 & (unsigned int)v119)];
              v44 = *v43;
              if ( v39 == *v43 )
              {
LABEL_45:
                if ( *((_BYTE *)v43 + 8) )
                {
                  v45 = *(_QWORD *)(a1 + 72);
                  v46 = *(unsigned int *)(a1 + 88);
                  if ( (_DWORD)v46 )
                  {
                    v47 = 1;
                    for ( i = (v46 - 1)
                            & (((0xBF58476D1CE4E5B9LL
                               * (v113
                                | ((unsigned __int64)(((unsigned int)&qword_4F8A320 >> 9)
                                                    ^ ((unsigned int)&qword_4F8A320 >> 4)) << 32))) >> 31)
                             ^ (484763065 * v113)); ; i = (v46 - 1) & v50 )
                    {
                      v49 = v45 + 24LL * i;
                      if ( *(__int64 **)v49 == &qword_4F8A320 && a2 == *(_QWORD *)(v49 + 8) )
                        break;
                      if ( *(_QWORD *)v49 == -4096 && *(_QWORD *)(v49 + 8) == -4096 )
                        goto LABEL_90;
                      v50 = v47 + i;
                      ++v47;
                    }
                    if ( v49 != v45 + 24 * v46 )
                    {
                      v53 = *(_QWORD *)(*(_QWORD *)(v49 + 16) + 24LL);
                      if ( v53 )
                      {
                        v54 = *(unsigned int *)(a1 + 24);
                        v55 = *(_QWORD *)(a1 + 8);
                        if ( (_DWORD)v54 )
                        {
                          v56 = (v54 - 1) & (unsigned int)v119;
                          v57 = (__int64 *)(v55 + 16LL * v56);
                          v58 = *v57;
                          if ( v39 == *v57 )
                            goto LABEL_77;
                          v77 = 1;
                          while ( v58 != -4096 )
                          {
                            v97 = v77 + 1;
                            v56 = (v54 - 1) & (v77 + v56);
                            v57 = (__int64 *)(v55 + 16LL * v56);
                            v58 = *v57;
                            if ( v39 == *v57 )
                              goto LABEL_77;
                            v77 = v97;
                          }
                        }
                        v57 = (__int64 *)(v55 + 16 * v54);
LABEL_77:
                        v59 = v57[1];
                        v60 = *(_QWORD *)(v53 + 8);
                        if ( v60 )
                        {
                          v61 = *(_QWORD **)(v60 + 1008);
                          v62 = &v61[4 * *(unsigned int *)(v60 + 1016)];
                          if ( v61 != v62 )
                          {
                            v115 = *(_QWORD *)(v10 + 16);
                            v63 = v62;
                            do
                            {
                              v122 = 0;
                              v64 = (_QWORD *)sub_22077B0(0x10u);
                              if ( v64 )
                              {
                                v64[1] = a2;
                                *v64 = &unk_4A29888;
                              }
                              v65 = v122;
                              v122 = v64;
                              if ( v65 )
                                (*(void (__fastcall **)(_QWORD *))(*v65 + 8LL))(v65);
                              v66 = v61;
                              v68 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v59 + 24LL))(v59);
                              if ( (v61[3] & 2) == 0 )
                                v66 = (_QWORD *)*v61;
                              (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD **))(v61[3]
                                                                                            & 0xFFFFFFFFFFFFFFF8LL))(
                                v66,
                                v68,
                                v67,
                                &v122);
                              if ( v122 )
                                (*(void (__fastcall **)(_QWORD *))(*v122 + 8LL))(v122);
                              v61 += 4;
                            }
                            while ( v63 != v61 );
                            v39 = v115;
                          }
                        }
                      }
                    }
                  }
LABEL_90:
                  v69 = *(_QWORD *)v10;
                  --v117[3];
                  sub_2208CA0((__int64 *)v10);
                  v70 = *(_QWORD *)(v10 + 24);
                  if ( v70 )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v70 + 8LL))(v70);
                  j_j___libc_free_0(v10);
                  v71 = *(_DWORD *)(a1 + 88);
                  if ( v71 )
                  {
                    v72 = 1;
                    v73 = v71 - 1;
                    for ( j = (v71 - 1)
                            & (((0xBF58476D1CE4E5B9LL * (v113 | ((_QWORD)v119 << 32))) >> 31)
                             ^ (484763065 * v113)); ; j = v73 & v76 )
                    {
                      v75 = (_QWORD *)(*(_QWORD *)(a1 + 72) + 24LL * j);
                      if ( v39 == *v75 && a2 == v75[1] )
                        break;
                      if ( *v75 == -4096 && v75[1] == -4096 )
                        goto LABEL_99;
                      v76 = v72 + j;
                      ++v72;
                    }
                    *v75 = -8192;
                    v75[1] = -8192;
                    --*(_DWORD *)(a1 + 80);
                    ++*(_DWORD *)(a1 + 84);
                  }
LABEL_99:
                  v10 = v69;
                  goto LABEL_54;
                }
              }
              else
              {
                v51 = 1;
                while ( v44 != -4096 )
                {
                  v96 = v51 + 1;
                  v42 = v41 & (v51 + v42);
                  v43 = &v40[2 * v42];
                  v44 = *v43;
                  if ( v39 == *v43 )
                    goto LABEL_45;
                  v51 = v96;
                }
              }
LABEL_53:
              v10 = *(_QWORD *)v10;
LABEL_54:
              if ( v120 == (_QWORD *)v10 )
              {
                result = (__int64)v117;
                v10 = v117[1];
                goto LABEL_56;
              }
            }
          }
          goto LABEL_123;
        }
      }
    }
    goto LABEL_122;
  }
  v80 = 1;
  v81 = 0;
  while ( v9 != -4096 )
  {
    if ( !v81 && v9 == -8192 )
      v81 = v117;
    v8 = (v5 - 1) & (v80 + v8);
    v117 = (__int64 *)(v7 + 32LL * v8);
    v9 = *v117;
    if ( a2 == *v117 )
      goto LABEL_6;
    ++v80;
  }
  if ( !v81 )
    v81 = v117;
  ++*(_QWORD *)(a1 + 32);
  v82 = *(_DWORD *)(a1 + 48) + 1;
  if ( 4 * v82 >= 3 * v5 )
  {
LABEL_154:
    sub_2EB09A0(v6, 2 * v5);
    v98 = *(_DWORD *)(a1 + 56);
    if ( v98 )
    {
      v99 = v98 - 1;
      v100 = *(_QWORD *)(a1 + 40);
      v82 = *(_DWORD *)(a1 + 48) + 1;
      v101 = (v98 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v81 = (__int64 *)(v100 + 32LL * v101);
      v102 = *v81;
      if ( a2 != *v81 )
      {
        v103 = 1;
        v104 = 0;
        while ( v102 != -4096 )
        {
          if ( !v104 && v102 == -8192 )
            v104 = v81;
          v101 = v99 & (v103 + v101);
          v81 = (__int64 *)(v100 + 32LL * v101);
          v102 = *v81;
          if ( a2 == *v81 )
            goto LABEL_119;
          ++v103;
        }
        if ( v104 )
          v81 = v104;
      }
      goto LABEL_119;
    }
    goto LABEL_203;
  }
  if ( v5 - *(_DWORD *)(a1 + 52) - v82 <= v5 >> 3 )
  {
    sub_2EB09A0(v6, v5);
    v105 = *(_DWORD *)(a1 + 56);
    if ( v105 )
    {
      v106 = v105 - 1;
      v107 = *(_QWORD *)(a1 + 40);
      v108 = 1;
      v109 = (v105 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v110 = 0;
      v82 = *(_DWORD *)(a1 + 48) + 1;
      v81 = (__int64 *)(v107 + 32LL * v109);
      v111 = *v81;
      if ( a2 != *v81 )
      {
        while ( v111 != -4096 )
        {
          if ( !v110 && v111 == -8192 )
            v110 = v81;
          v109 = v106 & (v108 + v109);
          v81 = (__int64 *)(v107 + 32LL * v109);
          v111 = *v81;
          if ( a2 == *v81 )
            goto LABEL_119;
          ++v108;
        }
        if ( v110 )
          v81 = v110;
      }
      goto LABEL_119;
    }
LABEL_203:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
LABEL_119:
  *(_DWORD *)(a1 + 48) = v82;
  if ( *v81 != -4096 )
    --*(_DWORD *)(a1 + 52);
  v10 = (unsigned __int64)(v81 + 1);
  v81[3] = 0;
  v81[2] = (__int64)(v81 + 1);
  *v81 = a2;
  v81[1] = (__int64)(v81 + 1);
  v120 = v81 + 1;
LABEL_122:
  result = (unsigned int)v125 >> 1;
  if ( (unsigned int)v125 >> 1 )
    goto LABEL_123;
LABEL_56:
  if ( (_QWORD *)v10 == v120 )
  {
LABEL_123:
    v83 = *(_QWORD *)(a1 + 40);
    result = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)result )
    {
      v84 = result - 1;
      result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v85 = (__int64 *)(v83 + 32 * result);
      v86 = *v85;
      if ( a2 == *v85 )
      {
LABEL_125:
        v87 = (__int64 *)v85[1];
        while ( v85 + 1 != v87 )
        {
          v88 = (unsigned __int64)v87;
          v87 = (__int64 *)*v87;
          v89 = *(_QWORD *)(v88 + 24);
          if ( v89 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v89 + 8LL))(v89);
          j_j___libc_free_0(v88);
        }
        result = a1;
        *v85 = -8192;
        --*(_DWORD *)(a1 + 48);
        ++*(_DWORD *)(a1 + 52);
      }
      else
      {
        v112 = 1;
        while ( v86 != -4096 )
        {
          result = v84 & (unsigned int)(v112 + result);
          v85 = (__int64 *)(v83 + 32LL * (unsigned int)result);
          v86 = *v85;
          if ( a2 == *v85 )
            goto LABEL_125;
          ++v112;
        }
      }
    }
  }
  if ( (v125 & 1) == 0 )
    return sub_C7D6A0((__int64)v126, 16LL * v127, 8);
  return result;
}

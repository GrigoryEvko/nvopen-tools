// Function: sub_BBE020
// Address: 0xbbe020
//
__int64 __fastcall sub_BBE020(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char *v5; // rax
  __int64 v6; // r8
  unsigned int v7; // esi
  __int64 v8; // rdi
  unsigned int v9; // ecx
  __int64 v10; // rdx
  _QWORD *v11; // r15
  _QWORD *v12; // r9
  int v13; // esi
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r10
  __int64 v17; // rdx
  char v18; // al
  __int64 v19; // r14
  char v20; // cl
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 result; // rax
  __int64 v24; // r13
  _QWORD *v25; // rsi
  int v26; // edx
  unsigned int v27; // edi
  _QWORD *v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rcx
  __int64 v31; // rax
  int v32; // r9d
  unsigned int i; // edi
  __int64 v34; // rdx
  unsigned int v35; // edi
  int v36; // eax
  int v37; // eax
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rsi
  unsigned int v41; // edi
  __int64 *v42; // rax
  __int64 v43; // r10
  __int64 v44; // r12
  __int64 v45; // rax
  _QWORD *v46; // rbx
  _QWORD *v47; // rax
  _QWORD *v48; // r13
  _QWORD *v49; // rax
  __int64 v50; // rdi
  _QWORD *v51; // rdi
  __int64 v52; // rdx
  __int64 v53; // rsi
  _QWORD *v54; // rbx
  __int64 v55; // rdi
  int v56; // edx
  int v57; // esi
  unsigned int j; // eax
  _QWORD *v59; // rdx
  int v60; // eax
  int v61; // eax
  void **v62; // rdx
  void **v63; // rcx
  __int64 v64; // rdx
  __int64 v65; // rcx
  int v66; // r11d
  _QWORD *v67; // rax
  int v68; // ecx
  __int64 v69; // rcx
  __int64 v70; // rdx
  __int64 *v71; // r12
  __int64 v72; // rsi
  __int64 *v73; // rbx
  __int64 *v74; // r14
  __int64 v75; // rdi
  int v76; // r9d
  int v77; // eax
  int v78; // esi
  unsigned int v79; // edx
  __int64 v80; // rdi
  int v81; // r10d
  _QWORD *v82; // r9
  int v83; // eax
  int v84; // edx
  __int64 v85; // rdi
  int v86; // r9d
  unsigned int v87; // r12d
  __int64 v88; // rsi
  int v89; // edi
  __int64 v90; // [rsp+8h] [rbp-158h]
  __int64 v91; // [rsp+18h] [rbp-148h]
  _QWORD *v92; // [rsp+20h] [rbp-140h]
  __int64 v93; // [rsp+28h] [rbp-138h]
  _QWORD *v94; // [rsp+38h] [rbp-128h]
  _QWORD v97[2]; // [rsp+50h] [rbp-110h] BYREF
  __int64 v98; // [rsp+60h] [rbp-100h] BYREF
  char v99[8]; // [rsp+68h] [rbp-F8h] BYREF
  _QWORD v100[6]; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v101; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v102; // [rsp+A8h] [rbp-B8h]
  _QWORD *v103; // [rsp+B0h] [rbp-B0h] BYREF
  unsigned int v104; // [rsp+B8h] [rbp-A8h]
  char v105; // [rsp+130h] [rbp-30h] BYREF

  if ( *(_DWORD *)(a3 + 72) == *(_DWORD *)(a3 + 68) )
  {
    if ( *(_BYTE *)(a3 + 28) )
    {
      v62 = *(void ***)(a3 + 8);
      v63 = &v62[*(unsigned int *)(a3 + 20)];
      if ( v62 != v63 )
      {
        result = (__int64)v62;
        while ( *(_UNKNOWN **)result != &unk_4F82400 )
        {
          result += 8;
          if ( v63 == (void **)result )
            goto LABEL_127;
        }
        return result;
      }
    }
    else
    {
      result = sub_C8CA60(a3, &unk_4F82400, a3, a4);
      if ( result )
        return result;
      if ( *(_BYTE *)(a3 + 28) )
      {
        v62 = *(void ***)(a3 + 8);
        result = (__int64)&v62[*(unsigned int *)(a3 + 20)];
        if ( (void **)result != v62 )
        {
LABEL_127:
          while ( *v62 != &unk_4F82420 )
          {
            if ( ++v62 == (void **)result )
              goto LABEL_2;
          }
          return result;
        }
      }
      else
      {
        result = sub_C8CA60(a3, &unk_4F82420, v64, v65);
        if ( result )
          return result;
      }
    }
  }
LABEL_2:
  v5 = (char *)&v103;
  v101 = 0;
  v102 = 1;
  do
  {
    *(_QWORD *)v5 = -4096;
    v5 += 16;
  }
  while ( v5 != &v105 );
  v97[0] = &v101;
  v6 = a1 + 32;
  v97[1] = a1 + 64;
  v7 = *(_DWORD *)(a1 + 56);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_117;
  }
  v8 = *(_QWORD *)(a1 + 40);
  v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = *(_QWORD *)(v8 + 32LL * v9);
  v92 = (_QWORD *)(v8 + 32LL * v9);
  if ( a2 == v10 )
  {
LABEL_6:
    v11 = (_QWORD *)v92[1];
    v94 = v92 + 1;
    if ( v92 + 1 != v11 )
    {
      while ( 1 )
      {
        v19 = v11[2];
        v20 = v102 & 1;
        if ( (v102 & 1) != 0 )
        {
          v12 = &v103;
          v13 = 7;
        }
        else
        {
          v21 = v104;
          v12 = v103;
          if ( !v104 )
            goto LABEL_19;
          v13 = v104 - 1;
        }
        v14 = v13 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v15 = &v12[2 * v14];
        v16 = *v15;
        if ( v19 != *v15 )
          break;
LABEL_10:
        v17 = 16;
        if ( !v20 )
          v17 = 2LL * v104;
        if ( v15 == &v12[v17] )
        {
          v18 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD *))(*(_QWORD *)v11[3] + 16LL))(
                  v11[3],
                  a2,
                  a3,
                  v97);
          v98 = v19;
          v99[0] = v18;
          sub_BBCF50((__int64)v100, (__int64)&v101, &v98, v99);
        }
        v11 = (_QWORD *)*v11;
        if ( v11 == v94 )
        {
          v11 = (_QWORD *)v92[1];
          result = (unsigned int)v102 >> 1;
          if ( !((unsigned int)v102 >> 1) )
            goto LABEL_38;
          if ( v94 != v11 )
          {
            v90 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
            while ( 1 )
            {
              v24 = v11[2];
              if ( (v102 & 1) != 0 )
              {
                v25 = &v103;
                v26 = 7;
              }
              else
              {
                v25 = v103;
                if ( !v104 )
                  goto LABEL_35;
                v26 = v104 - 1;
              }
              LODWORD(v93) = ((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4);
              v27 = v26 & v93;
              v28 = &v25[2 * (v26 & (unsigned int)v93)];
              v29 = *v28;
              if ( v24 == *v28 )
              {
LABEL_27:
                if ( *((_BYTE *)v28 + 8) )
                {
                  v30 = *(_QWORD *)(a1 + 72);
                  v31 = *(unsigned int *)(a1 + 88);
                  if ( (_DWORD)v31 )
                  {
                    v32 = 1;
                    for ( i = (v31 - 1)
                            & (((0xBF58476D1CE4E5B9LL
                               * (v90
                                | ((unsigned __int64)(((unsigned int)&unk_4F8A320 >> 9)
                                                    ^ ((unsigned int)&unk_4F8A320 >> 4)) << 32))) >> 31)
                             ^ (484763065 * v90)); ; i = (v31 - 1) & v35 )
                    {
                      v34 = v30 + 24LL * i;
                      if ( *(_UNKNOWN **)v34 == &unk_4F8A320 && a2 == *(_QWORD *)(v34 + 8) )
                        break;
                      if ( *(_QWORD *)v34 == -4096 && *(_QWORD *)(v34 + 8) == -4096 )
                        goto LABEL_66;
                      v35 = v32 + i;
                      ++v32;
                    }
                    if ( v34 != v30 + 24 * v31 )
                    {
                      v38 = *(_QWORD *)(*(_QWORD *)(v34 + 16) + 24LL);
                      if ( v38 )
                      {
                        v39 = *(unsigned int *)(a1 + 24);
                        v40 = *(_QWORD *)(a1 + 8);
                        if ( (_DWORD)v39 )
                        {
                          v41 = (v39 - 1) & v93;
                          v42 = (__int64 *)(v40 + 16LL * v41);
                          v43 = *v42;
                          if ( v24 == *v42 )
                            goto LABEL_53;
                          v61 = 1;
                          while ( v43 != -4096 )
                          {
                            v76 = v61 + 1;
                            v41 = (v39 - 1) & (v61 + v41);
                            v42 = (__int64 *)(v40 + 16LL * v41);
                            v43 = *v42;
                            if ( v24 == *v42 )
                              goto LABEL_53;
                            v61 = v76;
                          }
                        }
                        v42 = (__int64 *)(v40 + 16 * v39);
LABEL_53:
                        v44 = v42[1];
                        v45 = *(_QWORD *)(v38 + 8);
                        if ( v45 )
                        {
                          v46 = *(_QWORD **)(v45 + 1008);
                          v47 = &v46[4 * *(unsigned int *)(v45 + 1016)];
                          if ( v46 != v47 )
                          {
                            v91 = v11[2];
                            v48 = v47;
                            do
                            {
                              v100[0] = 0;
                              v49 = (_QWORD *)sub_22077B0(16);
                              if ( v49 )
                              {
                                v49[1] = a2;
                                *v49 = &unk_49DB0A8;
                              }
                              v50 = v100[0];
                              v100[0] = v49;
                              if ( v50 )
                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v50 + 8LL))(v50);
                              v51 = v46;
                              v53 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v44 + 24LL))(v44);
                              if ( (v46[3] & 2) == 0 )
                                v51 = (_QWORD *)*v46;
                              (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD *))(v46[3]
                                                                                           & 0xFFFFFFFFFFFFFFF8LL))(
                                v51,
                                v53,
                                v52,
                                v100);
                              if ( v100[0] )
                                (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v100[0] + 8LL))(v100[0]);
                              v46 += 4;
                            }
                            while ( v48 != v46 );
                            v24 = v91;
                          }
                        }
                      }
                    }
                  }
LABEL_66:
                  v54 = (_QWORD *)*v11;
                  --v92[3];
                  sub_2208CA0(v11);
                  v55 = v11[3];
                  if ( v55 )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v55 + 8LL))(v55);
                  j_j___libc_free_0(v11, 32);
                  v56 = *(_DWORD *)(a1 + 88);
                  if ( v56 )
                  {
                    v6 = 1;
                    v57 = v56 - 1;
                    for ( j = (v56 - 1) & (((0xBF58476D1CE4E5B9LL * (v90 | (v93 << 32))) >> 31) ^ (484763065 * v90));
                          ;
                          j = v57 & v60 )
                    {
                      v59 = (_QWORD *)(*(_QWORD *)(a1 + 72) + 24LL * j);
                      if ( v24 == *v59 && a2 == v59[1] )
                        break;
                      if ( *v59 == -4096 && v59[1] == -4096 )
                        goto LABEL_75;
                      v60 = v6 + j;
                      v6 = (unsigned int)(v6 + 1);
                    }
                    *v59 = -8192;
                    v59[1] = -8192;
                    --*(_DWORD *)(a1 + 80);
                    ++*(_DWORD *)(a1 + 84);
                  }
LABEL_75:
                  v11 = v54;
                  goto LABEL_36;
                }
              }
              else
              {
                v36 = 1;
                while ( v29 != -4096 )
                {
                  v6 = (unsigned int)(v36 + 1);
                  v27 = v26 & (v36 + v27);
                  v28 = &v25[2 * v27];
                  v29 = *v28;
                  if ( v24 == *v28 )
                    goto LABEL_27;
                  v36 = v6;
                }
              }
LABEL_35:
              v11 = (_QWORD *)*v11;
LABEL_36:
              if ( v94 == v11 )
              {
                result = (__int64)v92;
                v11 = (_QWORD *)v92[1];
                goto LABEL_38;
              }
            }
          }
          goto LABEL_99;
        }
      }
      v37 = 1;
      while ( v16 != -4096 )
      {
        v6 = (unsigned int)(v37 + 1);
        v14 = v13 & (v37 + v14);
        v15 = &v12[2 * v14];
        v16 = *v15;
        if ( v19 == *v15 )
          goto LABEL_10;
        v37 = v6;
      }
      if ( v20 )
      {
        v22 = 16;
      }
      else
      {
        v21 = v104;
LABEL_19:
        v22 = 2 * v21;
      }
      v15 = &v12[v22];
      goto LABEL_10;
    }
    goto LABEL_98;
  }
  v66 = 1;
  v67 = 0;
  while ( v10 != -4096 )
  {
    if ( !v67 && v10 == -8192 )
      v67 = v92;
    v9 = (v7 - 1) & (v66 + v9);
    v92 = (_QWORD *)(v8 + 32LL * v9);
    v10 = *v92;
    if ( a2 == *v92 )
      goto LABEL_6;
    ++v66;
  }
  if ( !v67 )
    v67 = v92;
  ++*(_QWORD *)(a1 + 32);
  v68 = *(_DWORD *)(a1 + 48) + 1;
  if ( 4 * v68 >= 3 * v7 )
  {
LABEL_117:
    sub_BBC880(v6, 2 * v7);
    v77 = *(_DWORD *)(a1 + 56);
    if ( v77 )
    {
      v78 = v77 - 1;
      v6 = *(_QWORD *)(a1 + 40);
      v68 = *(_DWORD *)(a1 + 48) + 1;
      v79 = (v77 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v67 = (_QWORD *)(v6 + 32LL * v79);
      v80 = *v67;
      if ( a2 != *v67 )
      {
        v81 = 1;
        v82 = 0;
        while ( v80 != -4096 )
        {
          if ( !v82 && v80 == -8192 )
            v82 = v67;
          v79 = v78 & (v81 + v79);
          v67 = (_QWORD *)(v6 + 32LL * v79);
          v80 = *v67;
          if ( a2 == *v67 )
            goto LABEL_95;
          ++v81;
        }
        if ( v82 )
          v67 = v82;
      }
      goto LABEL_95;
    }
    goto LABEL_154;
  }
  if ( v7 - *(_DWORD *)(a1 + 52) - v68 <= v7 >> 3 )
  {
    sub_BBC880(v6, v7);
    v83 = *(_DWORD *)(a1 + 56);
    if ( v83 )
    {
      v84 = v83 - 1;
      v85 = *(_QWORD *)(a1 + 40);
      v86 = 1;
      v87 = (v83 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v6 = 0;
      v68 = *(_DWORD *)(a1 + 48) + 1;
      v67 = (_QWORD *)(v85 + 32LL * v87);
      v88 = *v67;
      if ( a2 != *v67 )
      {
        while ( v88 != -4096 )
        {
          if ( !v6 && v88 == -8192 )
            v6 = (__int64)v67;
          v87 = v84 & (v86 + v87);
          v67 = (_QWORD *)(v85 + 32LL * v87);
          v88 = *v67;
          if ( a2 == *v67 )
            goto LABEL_95;
          ++v86;
        }
        if ( v6 )
          v67 = (_QWORD *)v6;
      }
      goto LABEL_95;
    }
LABEL_154:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
LABEL_95:
  *(_DWORD *)(a1 + 48) = v68;
  if ( *v67 != -4096 )
    --*(_DWORD *)(a1 + 52);
  v11 = v67 + 1;
  v67[3] = 0;
  v67[2] = v67 + 1;
  *v67 = a2;
  v67[1] = v67 + 1;
  v94 = v67 + 1;
LABEL_98:
  result = (unsigned int)v102 >> 1;
  if ( (unsigned int)v102 >> 1 )
    goto LABEL_99;
LABEL_38:
  if ( v11 == v94 )
  {
LABEL_99:
    v69 = *(_QWORD *)(a1 + 40);
    result = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)result )
    {
      v70 = (unsigned int)(result - 1);
      result = (unsigned int)v70 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v71 = (__int64 *)(v69 + 32 * result);
      v72 = *v71;
      if ( a2 == *v71 )
      {
LABEL_101:
        v73 = (__int64 *)v71[1];
        while ( v71 + 1 != v73 )
        {
          v74 = v73;
          v73 = (__int64 *)*v73;
          v75 = v74[3];
          if ( v75 )
            (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v75 + 8LL))(
              v75,
              v72,
              v70,
              v69,
              v6);
          v72 = 32;
          j_j___libc_free_0(v74, 32);
        }
        result = a1;
        *v71 = -8192;
        --*(_DWORD *)(a1 + 48);
        ++*(_DWORD *)(a1 + 52);
      }
      else
      {
        v89 = 1;
        while ( v72 != -4096 )
        {
          v6 = (unsigned int)(v89 + 1);
          result = (unsigned int)v70 & (v89 + (_DWORD)result);
          v71 = (__int64 *)(v69 + 32LL * (unsigned int)result);
          v72 = *v71;
          if ( a2 == *v71 )
            goto LABEL_101;
          ++v89;
        }
      }
    }
  }
  if ( (v102 & 1) == 0 )
    return sub_C7D6A0(v103, 16LL * v104, 8);
  return result;
}

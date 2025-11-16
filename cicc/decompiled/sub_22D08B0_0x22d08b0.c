// Function: sub_22D08B0
// Address: 0x22d08b0
//
__int64 __fastcall sub_22D08B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rax
  __int64 v5; // r8
  unsigned int v6; // esi
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 v9; // rdx
  unsigned __int64 v10; // r15
  __int64 *v11; // r9
  int v12; // esi
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r10
  __int64 v16; // rdx
  char v17; // al
  __int64 v18; // r14
  char v19; // cl
  __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 result; // rax
  __int64 v23; // r13
  __int64 *v24; // rsi
  int v25; // edx
  unsigned int v26; // edi
  __int64 *v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rcx
  __int64 v30; // rax
  int v31; // r9d
  unsigned int i; // edi
  __int64 v33; // rdx
  unsigned int v34; // edi
  int v35; // eax
  int v36; // eax
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rsi
  unsigned int v40; // edi
  __int64 *v41; // rax
  __int64 v42; // r10
  __int64 v43; // r12
  __int64 v44; // rax
  _QWORD *v45; // rbx
  _QWORD *v46; // rax
  _QWORD *v47; // r13
  _QWORD *v48; // rax
  __int64 v49; // rdi
  _QWORD *v50; // rdi
  __int64 v51; // rdx
  __int64 v52; // rsi
  unsigned __int64 v53; // rbx
  __int64 v54; // rdi
  int v55; // edx
  int v56; // esi
  unsigned int j; // eax
  _QWORD *v58; // rdx
  int v59; // eax
  int v60; // eax
  void **v61; // rdx
  __int64 *v62; // rcx
  int v63; // r11d
  _QWORD *v64; // rax
  int v65; // ecx
  __int64 v66; // rcx
  __int64 v67; // rdx
  __int64 *v68; // r12
  __int64 v69; // rsi
  __int64 *v70; // rbx
  unsigned __int64 v71; // r14
  __int64 v72; // rdi
  int v73; // r9d
  int v74; // eax
  int v75; // esi
  unsigned int v76; // edx
  __int64 v77; // rdi
  int v78; // r10d
  _QWORD *v79; // r9
  int v80; // eax
  int v81; // edx
  __int64 v82; // rdi
  int v83; // r9d
  unsigned int v84; // r12d
  __int64 v85; // rsi
  int v86; // edi
  __int64 v87; // [rsp+8h] [rbp-158h]
  __int64 v88; // [rsp+18h] [rbp-148h]
  __int64 *v89; // [rsp+20h] [rbp-140h]
  __int64 v90; // [rsp+28h] [rbp-138h]
  _QWORD *v91; // [rsp+38h] [rbp-128h]
  _QWORD v94[2]; // [rsp+50h] [rbp-110h] BYREF
  __int64 v95; // [rsp+60h] [rbp-100h] BYREF
  char v96[8]; // [rsp+68h] [rbp-F8h] BYREF
  _QWORD v97[6]; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v98; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v99; // [rsp+A8h] [rbp-B8h]
  __int64 *v100; // [rsp+B0h] [rbp-B0h] BYREF
  unsigned int v101; // [rsp+B8h] [rbp-A8h]
  char v102; // [rsp+130h] [rbp-30h] BYREF

  if ( *(_DWORD *)(a3 + 72) == *(_DWORD *)(a3 + 68) )
  {
    if ( *(_BYTE *)(a3 + 28) )
    {
      v61 = *(void ***)(a3 + 8);
      v62 = (__int64 *)&v61[*(unsigned int *)(a3 + 20)];
      if ( v61 != (void **)v62 )
      {
        result = (__int64)v61;
        while ( *(__int64 **)result != &qword_4F82400 )
        {
          result += 8;
          if ( v62 == (__int64 *)result )
            goto LABEL_127;
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
        v61 = *(void ***)(a3 + 8);
        result = (__int64)&v61[*(unsigned int *)(a3 + 20)];
        if ( (void **)result != v61 )
        {
LABEL_127:
          while ( *v61 != &unk_4FDBCE8 )
          {
            if ( ++v61 == (void **)result )
              goto LABEL_2;
          }
          return result;
        }
      }
      else
      {
        result = (__int64)sub_C8CA60(a3, (__int64)&unk_4FDBCE8);
        if ( result )
          return result;
      }
    }
  }
LABEL_2:
  v4 = (__int64 *)&v100;
  v98 = 0;
  v99 = 1;
  do
  {
    *v4 = -4096;
    v4 += 2;
  }
  while ( v4 != (__int64 *)&v102 );
  v94[0] = &v98;
  v5 = a1 + 32;
  v94[1] = a1 + 64;
  v6 = *(_DWORD *)(a1 + 56);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_117;
  }
  v7 = *(_QWORD *)(a1 + 40);
  v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = *(_QWORD *)(v7 + 32LL * v8);
  v89 = (__int64 *)(v7 + 32LL * v8);
  if ( a2 == v9 )
  {
LABEL_6:
    v10 = v89[1];
    v91 = v89 + 1;
    if ( v89 + 1 != (__int64 *)v10 )
    {
      while ( 1 )
      {
        v18 = *(_QWORD *)(v10 + 16);
        v19 = v99 & 1;
        if ( (v99 & 1) != 0 )
        {
          v11 = (__int64 *)&v100;
          v12 = 7;
        }
        else
        {
          v20 = v101;
          v11 = v100;
          if ( !v101 )
            goto LABEL_19;
          v12 = v101 - 1;
        }
        v13 = v12 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v14 = &v11[2 * v13];
        v15 = *v14;
        if ( v18 != *v14 )
          break;
LABEL_10:
        v16 = 16;
        if ( !v19 )
          v16 = 2LL * v101;
        if ( v14 == &v11[v16] )
        {
          v17 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD *))(**(_QWORD **)(v10 + 24) + 16LL))(
                  *(_QWORD *)(v10 + 24),
                  a2,
                  a3,
                  v94);
          v95 = v18;
          v96[0] = v17;
          sub_BBCF50((__int64)v97, (__int64)&v98, &v95, v96);
        }
        v10 = *(_QWORD *)v10;
        if ( (_QWORD *)v10 == v91 )
        {
          v10 = v89[1];
          result = (unsigned int)v99 >> 1;
          if ( !((unsigned int)v99 >> 1) )
            goto LABEL_38;
          if ( v91 != (_QWORD *)v10 )
          {
            v87 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
            while ( 1 )
            {
              v23 = *(_QWORD *)(v10 + 16);
              if ( (v99 & 1) != 0 )
              {
                v24 = (__int64 *)&v100;
                v25 = 7;
              }
              else
              {
                v24 = v100;
                if ( !v101 )
                  goto LABEL_35;
                v25 = v101 - 1;
              }
              LODWORD(v90) = ((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4);
              v26 = v25 & v90;
              v27 = &v24[2 * (v25 & (unsigned int)v90)];
              v28 = *v27;
              if ( v23 == *v27 )
              {
LABEL_27:
                if ( *((_BYTE *)v27 + 8) )
                {
                  v29 = *(_QWORD *)(a1 + 72);
                  v30 = *(unsigned int *)(a1 + 88);
                  if ( (_DWORD)v30 )
                  {
                    v31 = 1;
                    for ( i = (v30 - 1)
                            & (((0xBF58476D1CE4E5B9LL
                               * (v87
                                | ((unsigned __int64)(((unsigned int)&qword_4F8A320 >> 9)
                                                    ^ ((unsigned int)&qword_4F8A320 >> 4)) << 32))) >> 31)
                             ^ (484763065 * v87)); ; i = (v30 - 1) & v34 )
                    {
                      v33 = v29 + 24LL * i;
                      if ( *(__int64 **)v33 == &qword_4F8A320 && a2 == *(_QWORD *)(v33 + 8) )
                        break;
                      if ( *(_QWORD *)v33 == -4096 && *(_QWORD *)(v33 + 8) == -4096 )
                        goto LABEL_66;
                      v34 = v31 + i;
                      ++v31;
                    }
                    if ( v33 != v29 + 24 * v30 )
                    {
                      v37 = *(_QWORD *)(*(_QWORD *)(v33 + 16) + 24LL);
                      if ( v37 )
                      {
                        v38 = *(unsigned int *)(a1 + 24);
                        v39 = *(_QWORD *)(a1 + 8);
                        if ( (_DWORD)v38 )
                        {
                          v40 = (v38 - 1) & v90;
                          v41 = (__int64 *)(v39 + 16LL * v40);
                          v42 = *v41;
                          if ( v23 == *v41 )
                            goto LABEL_53;
                          v60 = 1;
                          while ( v42 != -4096 )
                          {
                            v73 = v60 + 1;
                            v40 = (v38 - 1) & (v60 + v40);
                            v41 = (__int64 *)(v39 + 16LL * v40);
                            v42 = *v41;
                            if ( v23 == *v41 )
                              goto LABEL_53;
                            v60 = v73;
                          }
                        }
                        v41 = (__int64 *)(v39 + 16 * v38);
LABEL_53:
                        v43 = v41[1];
                        v44 = *(_QWORD *)(v37 + 8);
                        if ( v44 )
                        {
                          v45 = *(_QWORD **)(v44 + 1008);
                          v46 = &v45[4 * *(unsigned int *)(v44 + 1016)];
                          if ( v45 != v46 )
                          {
                            v88 = *(_QWORD *)(v10 + 16);
                            v47 = v46;
                            do
                            {
                              v97[0] = 0;
                              v48 = (_QWORD *)sub_22077B0(0x10u);
                              if ( v48 )
                              {
                                v48[1] = a2;
                                *v48 = &unk_4A09EA8;
                              }
                              v49 = v97[0];
                              v97[0] = v48;
                              if ( v49 )
                                (*(void (__fastcall **)(__int64))(*(_QWORD *)v49 + 8LL))(v49);
                              v50 = v45;
                              v52 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v43 + 24LL))(v43);
                              if ( (v45[3] & 2) == 0 )
                                v50 = (_QWORD *)*v45;
                              (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD *))(v45[3]
                                                                                           & 0xFFFFFFFFFFFFFFF8LL))(
                                v50,
                                v52,
                                v51,
                                v97);
                              if ( v97[0] )
                                (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v97[0] + 8LL))(v97[0]);
                              v45 += 4;
                            }
                            while ( v47 != v45 );
                            v23 = v88;
                          }
                        }
                      }
                    }
                  }
LABEL_66:
                  v53 = *(_QWORD *)v10;
                  --v89[3];
                  sub_2208CA0((__int64 *)v10);
                  v54 = *(_QWORD *)(v10 + 24);
                  if ( v54 )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v54 + 8LL))(v54);
                  j_j___libc_free_0(v10);
                  v55 = *(_DWORD *)(a1 + 88);
                  if ( v55 )
                  {
                    v5 = 1;
                    v56 = v55 - 1;
                    for ( j = (v55 - 1) & (((0xBF58476D1CE4E5B9LL * (v87 | (v90 << 32))) >> 31) ^ (484763065 * v87));
                          ;
                          j = v56 & v59 )
                    {
                      v58 = (_QWORD *)(*(_QWORD *)(a1 + 72) + 24LL * j);
                      if ( v23 == *v58 && a2 == v58[1] )
                        break;
                      if ( *v58 == -4096 && v58[1] == -4096 )
                        goto LABEL_75;
                      v59 = v5 + j;
                      v5 = (unsigned int)(v5 + 1);
                    }
                    *v58 = -8192;
                    v58[1] = -8192;
                    --*(_DWORD *)(a1 + 80);
                    ++*(_DWORD *)(a1 + 84);
                  }
LABEL_75:
                  v10 = v53;
                  goto LABEL_36;
                }
              }
              else
              {
                v35 = 1;
                while ( v28 != -4096 )
                {
                  v5 = (unsigned int)(v35 + 1);
                  v26 = v25 & (v35 + v26);
                  v27 = &v24[2 * v26];
                  v28 = *v27;
                  if ( v23 == *v27 )
                    goto LABEL_27;
                  v35 = v5;
                }
              }
LABEL_35:
              v10 = *(_QWORD *)v10;
LABEL_36:
              if ( v91 == (_QWORD *)v10 )
              {
                result = (__int64)v89;
                v10 = v89[1];
                goto LABEL_38;
              }
            }
          }
          goto LABEL_99;
        }
      }
      v36 = 1;
      while ( v15 != -4096 )
      {
        v5 = (unsigned int)(v36 + 1);
        v13 = v12 & (v36 + v13);
        v14 = &v11[2 * v13];
        v15 = *v14;
        if ( v18 == *v14 )
          goto LABEL_10;
        v36 = v5;
      }
      if ( v19 )
      {
        v21 = 16;
      }
      else
      {
        v20 = v101;
LABEL_19:
        v21 = 2 * v20;
      }
      v14 = &v11[v21];
      goto LABEL_10;
    }
    goto LABEL_98;
  }
  v63 = 1;
  v64 = 0;
  while ( v9 != -4096 )
  {
    if ( !v64 && v9 == -8192 )
      v64 = v89;
    v8 = (v6 - 1) & (v63 + v8);
    v89 = (__int64 *)(v7 + 32LL * v8);
    v9 = *v89;
    if ( a2 == *v89 )
      goto LABEL_6;
    ++v63;
  }
  if ( !v64 )
    v64 = v89;
  ++*(_QWORD *)(a1 + 32);
  v65 = *(_DWORD *)(a1 + 48) + 1;
  if ( 4 * v65 >= 3 * v6 )
  {
LABEL_117:
    sub_22D0430(v5, 2 * v6);
    v74 = *(_DWORD *)(a1 + 56);
    if ( v74 )
    {
      v75 = v74 - 1;
      v5 = *(_QWORD *)(a1 + 40);
      v65 = *(_DWORD *)(a1 + 48) + 1;
      v76 = (v74 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v64 = (_QWORD *)(v5 + 32LL * v76);
      v77 = *v64;
      if ( a2 != *v64 )
      {
        v78 = 1;
        v79 = 0;
        while ( v77 != -4096 )
        {
          if ( !v79 && v77 == -8192 )
            v79 = v64;
          v76 = v75 & (v78 + v76);
          v64 = (_QWORD *)(v5 + 32LL * v76);
          v77 = *v64;
          if ( a2 == *v64 )
            goto LABEL_95;
          ++v78;
        }
        if ( v79 )
          v64 = v79;
      }
      goto LABEL_95;
    }
    goto LABEL_154;
  }
  if ( v6 - *(_DWORD *)(a1 + 52) - v65 <= v6 >> 3 )
  {
    sub_22D0430(v5, v6);
    v80 = *(_DWORD *)(a1 + 56);
    if ( v80 )
    {
      v81 = v80 - 1;
      v82 = *(_QWORD *)(a1 + 40);
      v83 = 1;
      v84 = (v80 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v5 = 0;
      v65 = *(_DWORD *)(a1 + 48) + 1;
      v64 = (_QWORD *)(v82 + 32LL * v84);
      v85 = *v64;
      if ( a2 != *v64 )
      {
        while ( v85 != -4096 )
        {
          if ( !v5 && v85 == -8192 )
            v5 = (__int64)v64;
          v84 = v81 & (v83 + v84);
          v64 = (_QWORD *)(v82 + 32LL * v84);
          v85 = *v64;
          if ( a2 == *v64 )
            goto LABEL_95;
          ++v83;
        }
        if ( v5 )
          v64 = (_QWORD *)v5;
      }
      goto LABEL_95;
    }
LABEL_154:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
LABEL_95:
  *(_DWORD *)(a1 + 48) = v65;
  if ( *v64 != -4096 )
    --*(_DWORD *)(a1 + 52);
  v10 = (unsigned __int64)(v64 + 1);
  v64[3] = 0;
  v64[2] = v64 + 1;
  *v64 = a2;
  v64[1] = v64 + 1;
  v91 = v64 + 1;
LABEL_98:
  result = (unsigned int)v99 >> 1;
  if ( (unsigned int)v99 >> 1 )
    goto LABEL_99;
LABEL_38:
  if ( (_QWORD *)v10 == v91 )
  {
LABEL_99:
    v66 = *(_QWORD *)(a1 + 40);
    result = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)result )
    {
      v67 = (unsigned int)(result - 1);
      result = (unsigned int)v67 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v68 = (__int64 *)(v66 + 32 * result);
      v69 = *v68;
      if ( a2 == *v68 )
      {
LABEL_101:
        v70 = (__int64 *)v68[1];
        while ( v68 + 1 != v70 )
        {
          v71 = (unsigned __int64)v70;
          v70 = (__int64 *)*v70;
          v72 = *(_QWORD *)(v71 + 24);
          if ( v72 )
            (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v72 + 8LL))(
              v72,
              v69,
              v67,
              v66,
              v5);
          v69 = 32;
          j_j___libc_free_0(v71);
        }
        result = a1;
        *v68 = -8192;
        --*(_DWORD *)(a1 + 48);
        ++*(_DWORD *)(a1 + 52);
      }
      else
      {
        v86 = 1;
        while ( v69 != -4096 )
        {
          v5 = (unsigned int)(v86 + 1);
          result = (unsigned int)v67 & (v86 + (_DWORD)result);
          v68 = (__int64 *)(v66 + 32LL * (unsigned int)result);
          v69 = *v68;
          if ( a2 == *v68 )
            goto LABEL_101;
          ++v86;
        }
      }
    }
  }
  if ( (v99 & 1) == 0 )
    return sub_C7D6A0((__int64)v100, 16LL * v101, 8);
  return result;
}

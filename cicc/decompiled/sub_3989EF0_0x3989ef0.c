// Function: sub_3989EF0
// Address: 0x3989ef0
//
__int64 __fastcall sub_3989EF0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rax
  unsigned __int8 *v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r12
  unsigned __int64 v8; // rcx
  unsigned int v9; // r8d
  __int64 *v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rbx
  unsigned int v14; // edx
  unsigned __int64 v15; // r13
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rcx
  __int64 v20; // rax
  __int64 *v21; // rsi
  _QWORD *k; // rax
  __int64 *m; // rax
  __int64 v24; // r11
  int v25; // ecx
  int v26; // ecx
  __int64 v27; // r9
  int v28; // r14d
  _QWORD *v29; // r10
  unsigned int v30; // edx
  _QWORD *v31; // rdi
  __int64 v32; // r8
  int v33; // edx
  int v34; // edi
  int v35; // esi
  unsigned int v36; // edx
  __int64 *v37; // rax
  __int64 v38; // r8
  int v39; // r9d
  int v40; // edi
  __int64 v41; // rax
  __int64 v42; // rax
  _QWORD *v43; // rcx
  __int64 v44; // rax
  __int64 *v45; // r10
  _QWORD *j; // rax
  __int64 *v47; // rax
  __int64 v48; // r11
  int v49; // ecx
  int v50; // ecx
  __int64 v51; // r8
  unsigned int v52; // edx
  _QWORD *v53; // rsi
  __int64 v54; // rdi
  int v55; // edx
  int v56; // edx
  int v57; // r9d
  __int64 *v58; // r8
  unsigned int v59; // r14d
  __int64 v60; // rsi
  __int64 v61; // rax
  _QWORD *v62; // rsi
  _QWORD *v63; // rax
  __int64 v64; // rax
  _QWORD *v65; // rsi
  _QWORD *v66; // rax
  _QWORD *v67; // r9
  int v68; // r10d
  __int64 *v69; // r9
  __int64 v70; // [rsp-70h] [rbp-70h]
  unsigned int v71; // [rsp-64h] [rbp-64h]
  unsigned int v72; // [rsp-64h] [rbp-64h]
  int v73; // [rsp-64h] [rbp-64h]
  __int64 v74; // [rsp-60h] [rbp-60h]
  __int64 i; // [rsp-58h] [rbp-58h]
  __int64 v76[8]; // [rsp-40h] [rbp-40h] BYREF

  result = (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL) + 504LL) - 34);
  if ( (unsigned int)result <= 1 && *(_DWORD *)(a1 + 6584) == 1 )
  {
    v3 = sub_1626D20(**(_QWORD **)(a1 + 4008));
    result = *(_QWORD *)(v3 + 8 * (5LL - *(unsigned int *)(v3 + 8)));
    if ( *(_DWORD *)(result + 36) == 3 )
    {
      v4 = (unsigned __int8 *)sub_16D40F0((__int64)qword_4FBB450);
      result = v4 ? *v4 : LOBYTE(qword_4FBB450[2]);
      if ( (_BYTE)result )
      {
        v5 = *(_QWORD *)(a1 + 4008);
        v6 = *(_QWORD *)(v5 + 328);
        result = v5 + 320;
        v70 = result;
        v74 = v6;
        if ( v6 != result )
        {
          do
          {
            v7 = *(_QWORD *)(v74 + 32);
            for ( i = v74 + 24; i != v7; v7 = *(_QWORD *)(v7 + 8) )
            {
              while ( 1 )
              {
                v12 = *(_QWORD *)(v7 + 64);
                v76[0] = v12;
                if ( v12 )
                {
                  sub_1623A60((__int64)v76, v12, 2);
                  if ( v76[0] )
                  {
                    v13 = sub_15C70F0((__int64)v76);
                    if ( v13 )
                    {
                      while ( 1 )
                      {
                        v14 = *(_DWORD *)(a1 + 6576);
                        v15 = *(_QWORD *)(a1 + 6560);
                        if ( !v14 )
                          break;
                        v8 = v14 - 1;
                        v9 = v8 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
                        v10 = (__int64 *)(v15 + 8LL * v9);
                        v11 = *v10;
                        if ( *v10 == v13 )
                          goto LABEL_13;
                        v39 = 1;
                        v37 = 0;
                        while ( v11 != -8 )
                        {
                          if ( v11 != -16 || v37 )
                            v10 = v37;
                          v9 = v8 & (v39 + v9);
                          v11 = *(_QWORD *)(v15 + 8LL * v9);
                          if ( v11 == v13 )
                            goto LABEL_13;
                          ++v39;
                          v37 = v10;
                          v10 = (__int64 *)(v15 + 8LL * v9);
                        }
                        if ( !v37 )
                          v37 = v10;
                        v40 = *(_DWORD *)(a1 + 6568);
                        ++*(_QWORD *)(a1 + 6552);
                        v34 = v40 + 1;
                        if ( 4 * v34 >= 3 * v14 )
                          goto LABEL_22;
                        if ( v14 - *(_DWORD *)(a1 + 6572) - v34 <= v14 >> 3 )
                        {
                          v72 = v14;
                          v41 = ((((((((v8 >> 1) | v8 | (((v8 >> 1) | v8) >> 2)) >> 4)
                                   | (v8 >> 1)
                                   | v8
                                   | (((v8 >> 1) | v8) >> 2)) >> 8)
                                 | (((v8 >> 1) | v8 | (((v8 >> 1) | v8) >> 2)) >> 4)
                                 | (v8 >> 1)
                                 | v8
                                 | (((v8 >> 1) | v8) >> 2)) >> 16)
                               | (((((v8 >> 1) | v8 | (((v8 >> 1) | v8) >> 2)) >> 4)
                                 | (v8 >> 1)
                                 | v8
                                 | (((v8 >> 1) | v8) >> 2)) >> 8)
                               | (((v8 >> 1) | v8 | (((v8 >> 1) | v8) >> 2)) >> 4)
                               | (v8 >> 1)
                               | v8
                               | (((v8 >> 1) | v8) >> 2))
                              + 1;
                          if ( (unsigned int)v41 < 0x40 )
                            LODWORD(v41) = 64;
                          *(_DWORD *)(a1 + 6576) = v41;
                          v42 = sub_22077B0(8LL * (unsigned int)v41);
                          *(_QWORD *)(a1 + 6560) = v42;
                          v43 = (_QWORD *)v42;
                          if ( v15 )
                          {
                            v44 = *(unsigned int *)(a1 + 6576);
                            v45 = (__int64 *)(v15 + 8LL * v72);
                            *(_QWORD *)(a1 + 6568) = 0;
                            for ( j = &v43[v44]; j != v43; ++v43 )
                            {
                              if ( v43 )
                                *v43 = -8;
                            }
                            v47 = (__int64 *)v15;
                            do
                            {
                              v48 = *v47;
                              if ( *v47 != -8 && v48 != -16 )
                              {
                                v49 = *(_DWORD *)(a1 + 6576);
                                if ( !v49 )
                                {
                                  MEMORY[0] = *v47;
                                  BUG();
                                }
                                v50 = v49 - 1;
                                v51 = *(_QWORD *)(a1 + 6560);
                                v52 = v50 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
                                v53 = (_QWORD *)(v51 + 8LL * v52);
                                v54 = *v53;
                                if ( v48 != *v53 )
                                {
                                  v73 = 1;
                                  v67 = 0;
                                  while ( v54 != -8 )
                                  {
                                    if ( !v67 && v54 == -16 )
                                      v67 = v53;
                                    v52 = v50 & (v73 + v52);
                                    v53 = (_QWORD *)(v51 + 8LL * v52);
                                    v54 = *v53;
                                    if ( v48 == *v53 )
                                      goto LABEL_68;
                                    ++v73;
                                  }
                                  if ( v67 )
                                    v53 = v67;
                                }
LABEL_68:
                                *v53 = v48;
                                ++*(_DWORD *)(a1 + 6568);
                              }
                              ++v47;
                            }
                            while ( v45 != v47 );
                            j___libc_free_0(v15);
                            v43 = *(_QWORD **)(a1 + 6560);
                            v55 = *(_DWORD *)(a1 + 6576);
                            v34 = *(_DWORD *)(a1 + 6568) + 1;
                          }
                          else
                          {
                            v64 = *(unsigned int *)(a1 + 6576);
                            *(_QWORD *)(a1 + 6568) = 0;
                            v65 = &v43[v64];
                            v55 = v64;
                            if ( v43 != v65 )
                            {
                              v66 = v43;
                              do
                              {
                                if ( v66 )
                                  *v66 = -8;
                                ++v66;
                              }
                              while ( v65 != v66 );
                            }
                            v34 = 1;
                          }
                          if ( !v55 )
                          {
LABEL_128:
                            ++*(_DWORD *)(a1 + 6568);
                            BUG();
                          }
                          v56 = v55 - 1;
                          v57 = 1;
                          v58 = 0;
                          v59 = v56 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
                          v37 = &v43[v59];
                          v60 = *v37;
                          if ( *v37 != v13 )
                          {
                            while ( v60 != -8 )
                            {
                              if ( v60 == -16 && !v58 )
                                v58 = v37;
                              v59 = v56 & (v57 + v59);
                              v37 = &v43[v59];
                              v60 = *v37;
                              if ( *v37 == v13 )
                                goto LABEL_39;
                              ++v57;
                            }
                            if ( v58 )
                              v37 = v58;
                          }
                        }
LABEL_39:
                        *(_DWORD *)(a1 + 6568) = v34;
                        if ( *v37 != -8 )
                          --*(_DWORD *)(a1 + 6572);
                        *v37 = v13;
                        if ( *(_DWORD *)(v13 + 8) == 2 )
                        {
                          v13 = *(_QWORD *)(v13 - 8);
                          if ( v13 )
                            continue;
                        }
                        goto LABEL_13;
                      }
                      ++*(_QWORD *)(a1 + 6552);
LABEL_22:
                      v71 = v14;
                      v16 = ((((((((2 * v14 - 1) | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 2)
                               | (2 * v14 - 1)
                               | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 4)
                             | (((2 * v14 - 1) | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 2)
                             | (2 * v14 - 1)
                             | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 8)
                           | (((((2 * v14 - 1) | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 2)
                             | (2 * v14 - 1)
                             | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 4)
                           | (((2 * v14 - 1) | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 2)
                           | (2 * v14 - 1)
                           | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 16;
                      v17 = (v16
                           | (((((((2 * v14 - 1) | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 2)
                               | (2 * v14 - 1)
                               | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 4)
                             | (((2 * v14 - 1) | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 2)
                             | (2 * v14 - 1)
                             | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 8)
                           | (((((2 * v14 - 1) | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 2)
                             | (2 * v14 - 1)
                             | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 4)
                           | (((2 * v14 - 1) | ((unsigned __int64)(2 * v14 - 1) >> 1)) >> 2)
                           | (2 * v14 - 1)
                           | ((unsigned __int64)(2 * v14 - 1) >> 1))
                          + 1;
                      if ( (unsigned int)v17 < 0x40 )
                        LODWORD(v17) = 64;
                      *(_DWORD *)(a1 + 6576) = v17;
                      v18 = sub_22077B0(8LL * (unsigned int)v17);
                      *(_QWORD *)(a1 + 6560) = v18;
                      v19 = (_QWORD *)v18;
                      if ( v15 )
                      {
                        v20 = *(unsigned int *)(a1 + 6576);
                        v21 = (__int64 *)(v15 + 8LL * v71);
                        *(_QWORD *)(a1 + 6568) = 0;
                        for ( k = &v19[v20]; k != v19; ++v19 )
                        {
                          if ( v19 )
                            *v19 = -8;
                        }
                        for ( m = (__int64 *)v15; v21 != m; ++m )
                        {
                          v24 = *m;
                          if ( *m != -8 && v24 != -16 )
                          {
                            v25 = *(_DWORD *)(a1 + 6576);
                            if ( !v25 )
                            {
                              MEMORY[0] = *m;
                              BUG();
                            }
                            v26 = v25 - 1;
                            v27 = *(_QWORD *)(a1 + 6560);
                            v28 = 1;
                            v29 = 0;
                            v30 = v26 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
                            v31 = (_QWORD *)(v27 + 8LL * v30);
                            v32 = *v31;
                            if ( *v31 != v24 )
                            {
                              while ( v32 != -8 )
                              {
                                if ( !v29 && v32 == -16 )
                                  v29 = v31;
                                v30 = v26 & (v28 + v30);
                                v31 = (_QWORD *)(v27 + 8LL * v30);
                                v32 = *v31;
                                if ( v24 == *v31 )
                                  goto LABEL_34;
                                ++v28;
                              }
                              if ( v29 )
                                v31 = v29;
                            }
LABEL_34:
                            *v31 = v24;
                            ++*(_DWORD *)(a1 + 6568);
                          }
                        }
                        j___libc_free_0(v15);
                        v19 = *(_QWORD **)(a1 + 6560);
                        v33 = *(_DWORD *)(a1 + 6576);
                        v34 = *(_DWORD *)(a1 + 6568) + 1;
                      }
                      else
                      {
                        v61 = *(unsigned int *)(a1 + 6576);
                        *(_QWORD *)(a1 + 6568) = 0;
                        v62 = &v19[v61];
                        v33 = v61;
                        if ( v19 != v62 )
                        {
                          v63 = v19;
                          do
                          {
                            if ( v63 )
                              *v63 = -8;
                            ++v63;
                          }
                          while ( v62 != v63 );
                        }
                        v34 = 1;
                      }
                      if ( !v33 )
                        goto LABEL_128;
                      v35 = v33 - 1;
                      v36 = (v33 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
                      v37 = &v19[v36];
                      v38 = *v37;
                      if ( *v37 != v13 )
                      {
                        v68 = 1;
                        v69 = 0;
                        while ( v38 != -8 )
                        {
                          if ( !v69 && v38 == -16 )
                            v69 = v37;
                          v36 = v35 & (v68 + v36);
                          v37 = &v19[v36];
                          v38 = *v37;
                          if ( *v37 == v13 )
                            goto LABEL_39;
                          ++v68;
                        }
                        if ( v69 )
                          v37 = v69;
                      }
                      goto LABEL_39;
                    }
LABEL_13:
                    if ( v76[0] )
                      sub_161E7C0((__int64)v76, v76[0]);
                  }
                }
                if ( (*(_BYTE *)v7 & 4) == 0 )
                  break;
                v7 = *(_QWORD *)(v7 + 8);
                if ( i == v7 )
                  goto LABEL_47;
              }
              while ( (*(_BYTE *)(v7 + 46) & 8) != 0 )
                v7 = *(_QWORD *)(v7 + 8);
            }
LABEL_47:
            result = *(_QWORD *)(v74 + 8);
            v74 = result;
          }
          while ( v70 != result );
        }
      }
    }
  }
  return result;
}

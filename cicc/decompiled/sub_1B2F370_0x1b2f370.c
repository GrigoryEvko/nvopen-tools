// Function: sub_1B2F370
// Address: 0x1b2f370
//
__int64 __fastcall sub_1B2F370(__int64 **a1, __int64 a2, char a3, char a4, __int64 a5)
{
  __int64 result; // rax
  __int64 *v6; // r14
  __int64 *v8; // r15
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 *v13; // r11
  unsigned int v14; // esi
  __int64 v15; // r8
  unsigned int v16; // edi
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rcx
  unsigned int i; // ecx
  _QWORD *v20; // rdx
  __int64 v21; // rdi
  unsigned int v22; // ecx
  int v23; // edi
  int v24; // edi
  int v25; // edi
  int v26; // r10d
  unsigned int v27; // esi
  __int64 v28; // r9
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // rdx
  int v32; // ecx
  unsigned int j; // edi
  __int64 v34; // rsi
  unsigned int v35; // edi
  int v36; // r10d
  int v37; // r10d
  int v38; // ecx
  unsigned int v39; // edi
  __int64 v40; // r9
  __int64 v41; // rsi
  unsigned int v42; // edi
  __int64 *v43; // [rsp+8h] [rbp-68h]
  unsigned __int64 v44; // [rsp+10h] [rbp-60h]
  __int64 *v45; // [rsp+10h] [rbp-60h]
  int v46; // [rsp+18h] [rbp-58h]
  __int64 v47; // [rsp+18h] [rbp-58h]
  __int64 v48; // [rsp+18h] [rbp-58h]
  _QWORD *v49; // [rsp+20h] [rbp-50h]
  bool v51; // [rsp+30h] [rbp-40h]
  __int64 v52; // [rsp+30h] [rbp-40h]

  result = **a1;
  v6 = (__int64 *)(result + 8LL * *((unsigned int *)*a1 + 2));
  if ( v6 != (__int64 *)result )
  {
    v8 = (__int64 *)**a1;
    do
    {
      result = (__int64)a1[1];
      v10 = *v8;
      if ( *v8 != *(_QWORD *)result )
      {
        result = *a1[2];
        if ( (v10 == result || !a3) && (v10 != result || !a4) )
        {
          v51 = v10 == result;
          v11 = sub_22077B0(72);
          if ( v11 )
          {
            v12 = *a1[1];
            *(_QWORD *)(v11 + 8) = 0;
            *(_QWORD *)(v11 + 16) = 0;
            *(_QWORD *)(v11 + 48) = v12;
            *(_DWORD *)(v11 + 24) = 0;
            *(_QWORD *)(v11 + 32) = a2;
            *(_QWORD *)(v11 + 40) = a5;
            *(_QWORD *)(v11 + 56) = v10;
            *(_QWORD *)v11 = &unk_49F6740;
            *(_BYTE *)(v11 + 64) = v51;
          }
          sub_1B2EEE0(a1[4], (__int64)a1[3], a2, v11);
          result = sub_157F0B0(v10);
          if ( !result )
          {
            v13 = a1[4];
            v14 = *((_DWORD *)v13 + 814);
            v15 = *a1[1];
            v52 = (__int64)(v13 + 404);
            if ( v14 )
            {
              v49 = 0;
              v46 = 1;
              v16 = (unsigned int)v10 >> 9;
              v17 = (((v16 ^ ((unsigned int)v10 >> 4)
                     | ((unsigned __int64)(((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)) << 32))
                    - 1
                    - ((unsigned __int64)(v16 ^ ((unsigned int)v10 >> 4)) << 32)) >> 22)
                  ^ ((v16 ^ ((unsigned int)v10 >> 4)
                    | ((unsigned __int64)(((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4)) << 32))
                   - 1
                   - ((unsigned __int64)(v16 ^ ((unsigned int)v10 >> 4)) << 32));
              v18 = 9 * (((v17 - 1 - (v17 << 13)) >> 8) ^ (v17 - 1 - (v17 << 13)));
              v44 = (((v18 ^ (v18 >> 15)) - 1 - ((v18 ^ (v18 >> 15)) << 27)) >> 31)
                  ^ ((v18 ^ (v18 >> 15)) - 1 - ((v18 ^ (v18 >> 15)) << 27));
              for ( i = (v14 - 1)
                      & ((((v18 ^ (v18 >> 15)) - 1 - ((v18 ^ (v18 >> 15)) << 27)) >> 31)
                       ^ ((v18 ^ (v18 >> 15)) - 1 - (((unsigned int)v18 ^ (unsigned int)(v18 >> 15)) << 27)));
                    ;
                    i = (v14 - 1) & v22 )
              {
                v20 = (_QWORD *)(v13[405] + 16LL * i);
                v21 = *v20;
                if ( v15 == *v20 && v10 == v20[1] )
                  goto LABEL_3;
                if ( v21 == -8 )
                {
                  if ( v20[1] == -8 )
                  {
                    if ( v49 )
                      v20 = v49;
                    v23 = *((_DWORD *)v13 + 812);
                    ++v13[404];
                    v24 = v23 + 1;
                    if ( 4 * v24 < 3 * v14 )
                    {
                      if ( v14 - *((_DWORD *)v13 + 813) - v24 > v14 >> 3 )
                      {
LABEL_29:
                        *((_DWORD *)v13 + 812) = v24;
                        if ( *v20 != -8 || v20[1] != -8 )
                          --*((_DWORD *)v13 + 813);
                        *v20 = v15;
                        v20[1] = v10;
                        goto LABEL_3;
                      }
                      v43 = v13;
                      v48 = v15;
                      sub_1A15100(v52, v14);
                      v13 = v43;
                      v36 = *((_DWORD *)v43 + 814);
                      if ( v36 )
                      {
                        v37 = v36 - 1;
                        v15 = v48;
                        v38 = 1;
                        v39 = v37 & v44;
                        v40 = v43[405];
                        result = 0;
                        while ( 1 )
                        {
                          v20 = (_QWORD *)(v40 + 16LL * v39);
                          v41 = *v20;
                          if ( v48 == *v20 && v10 == v20[1] )
                            break;
                          if ( v41 == -8 )
                          {
                            if ( v20[1] == -8 )
                              goto LABEL_58;
                          }
                          else if ( v41 == -16 && v20[1] == -16 && !result )
                          {
                            result = v40 + 16LL * v39;
                          }
                          v42 = v38 + v39;
                          ++v38;
                          v39 = v37 & v42;
                        }
LABEL_54:
                        result = *((unsigned int *)v13 + 812);
                        v24 = result + 1;
                        goto LABEL_29;
                      }
                      goto LABEL_63;
                    }
LABEL_35:
                    v45 = v13;
                    v47 = v15;
                    sub_1A15100(v52, 2 * v14);
                    v13 = v45;
                    v25 = *((_DWORD *)v45 + 814);
                    if ( v25 )
                    {
                      v15 = v47;
                      v26 = v25 - 1;
                      result = 0;
                      v27 = (unsigned int)v10 >> 9;
                      v28 = v45[405];
                      v29 = (((v27 ^ ((unsigned int)v10 >> 4)
                             | ((unsigned __int64)(((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4)) << 32))
                            - 1
                            - ((unsigned __int64)(v27 ^ ((unsigned int)v10 >> 4)) << 32)) >> 22)
                          ^ ((v27 ^ ((unsigned int)v10 >> 4)
                            | ((unsigned __int64)(((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4)) << 32))
                           - 1
                           - ((unsigned __int64)(v27 ^ ((unsigned int)v10 >> 4)) << 32));
                      v30 = 9 * (((v29 - 1 - (v29 << 13)) >> 8) ^ (v29 - 1 - (v29 << 13)));
                      v31 = (((v30 ^ (v30 >> 15)) - 1 - ((v30 ^ (v30 >> 15)) << 27)) >> 31)
                          ^ ((v30 ^ (v30 >> 15)) - 1 - ((v30 ^ (v30 >> 15)) << 27));
                      v32 = 1;
                      for ( j = (v25 - 1) & v31; ; j = v26 & v35 )
                      {
                        v20 = (_QWORD *)(v28 + 16LL * j);
                        v34 = *v20;
                        if ( v47 == *v20 && v10 == v20[1] )
                          break;
                        if ( v34 == -8 )
                        {
                          if ( v20[1] == -8 )
                          {
LABEL_58:
                            v24 = *((_DWORD *)v13 + 812) + 1;
                            if ( result )
                              v20 = (_QWORD *)result;
                            goto LABEL_29;
                          }
                        }
                        else if ( v34 == -16 && v20[1] == -16 && !result )
                        {
                          result = v28 + 16LL * j;
                        }
                        v35 = v32 + j;
                        ++v32;
                      }
                      goto LABEL_54;
                    }
LABEL_63:
                    ++*((_DWORD *)v13 + 812);
                    BUG();
                  }
                }
                else if ( v21 == -16 && v20[1] == -16 )
                {
                  if ( v49 )
                    v20 = v49;
                  v49 = v20;
                }
                v22 = v46 + i;
                ++v46;
              }
            }
            ++v13[404];
            goto LABEL_35;
          }
        }
      }
LABEL_3:
      ++v8;
    }
    while ( v6 != v8 );
  }
  return result;
}

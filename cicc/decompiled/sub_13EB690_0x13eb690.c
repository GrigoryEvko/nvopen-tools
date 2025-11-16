// Function: sub_13EB690
// Address: 0x13eb690
//
void __fastcall sub_13EB690(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // edx
  unsigned int v9; // edi
  __int64 *v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rcx
  int v14; // r9d
  unsigned int v15; // edi
  __int64 *v16; // r12
  __int64 v17; // rsi
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rax
  _QWORD *v20; // r12
  _QWORD *v21; // rbx
  __int64 v22; // r13
  __int64 v23; // rsi
  int v24; // ecx
  int v25; // r8d
  unsigned int v26; // edi
  __int64 *v27; // r14
  __int64 v28; // rax
  unsigned int v29; // eax
  int v30; // eax
  __int64 v31; // rdi
  __int64 v32; // rdi
  int v33; // ecx
  int v34; // r10d
  unsigned int v35; // [rsp-3Ch] [rbp-3Ch]
  unsigned int v36; // [rsp-3Ch] [rbp-3Ch]

  if ( a1[4] )
  {
    v3 = sub_157EB90(a2);
    v4 = sub_1632FA0(v3);
    v5 = sub_13E7A30(a1 + 4, *a1, v4, a1[3]);
    v6 = *(unsigned int *)(v5 + 24);
    if ( (_DWORD)v6 )
    {
      v7 = *(_QWORD *)(v5 + 8);
      v8 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
      v9 = (v6 - 1) & v8;
      v10 = (__int64 *)(v7 + 8LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
      {
LABEL_4:
        if ( v10 != (__int64 *)(v7 + 8 * v6) )
        {
          *v10 = -16;
          v12 = *(unsigned int *)(v5 + 88);
          --*(_DWORD *)(v5 + 16);
          ++*(_DWORD *)(v5 + 20);
          if ( (_DWORD)v12 )
          {
            v13 = *(_QWORD *)(v5 + 72);
            v14 = 1;
            v15 = (v12 - 1) & v8;
            v16 = (__int64 *)(v13 + 80LL * v15);
            v17 = *v16;
            if ( a2 == *v16 )
            {
LABEL_7:
              if ( v16 != (__int64 *)(v13 + 80 * v12) )
              {
                v18 = v16[3];
                if ( v18 != v16[2] )
                {
                  _libc_free(v18);
                  v8 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
                }
                *v16 = -16;
                --*(_DWORD *)(v5 + 80);
                ++*(_DWORD *)(v5 + 84);
              }
            }
            else
            {
              while ( v17 != -8 )
              {
                v15 = (v12 - 1) & (v14 + v15);
                v16 = (__int64 *)(v13 + 80LL * v15);
                v17 = *v16;
                if ( a2 == *v16 )
                  goto LABEL_7;
                ++v14;
              }
            }
          }
          if ( *(_DWORD *)(v5 + 48) )
          {
            v19 = *(_QWORD **)(v5 + 40);
            v20 = &v19[2 * *(unsigned int *)(v5 + 56)];
            if ( v19 != v20 )
            {
              while ( 1 )
              {
                v21 = v19;
                if ( *v19 != -8 && *v19 != -16 )
                  break;
                v19 += 2;
                if ( v20 == v19 )
                  return;
              }
              while ( 1 )
              {
                if ( v20 == v21 )
                  return;
                v22 = v21[1];
                if ( (*(_BYTE *)(v22 + 48) & 1) != 0 )
                {
                  v23 = v22 + 56;
                  v24 = 3;
                }
                else
                {
                  v30 = *(_DWORD *)(v22 + 64);
                  v23 = *(_QWORD *)(v22 + 56);
                  if ( !v30 )
                    goto LABEL_25;
                  v24 = v30 - 1;
                }
                v25 = 1;
                v26 = v24 & v8;
                v27 = (__int64 *)(v23 + 48LL * (v24 & v8));
                v28 = *v27;
                if ( a2 == *v27 )
                {
LABEL_23:
                  if ( *((_DWORD *)v27 + 2) == 3 )
                  {
                    if ( *((_DWORD *)v27 + 10) > 0x40u )
                    {
                      v31 = v27[4];
                      if ( v31 )
                      {
                        v35 = v8;
                        j_j___libc_free_0_0(v31);
                        v8 = v35;
                      }
                    }
                    if ( *((_DWORD *)v27 + 6) > 0x40u )
                    {
                      v32 = v27[2];
                      if ( v32 )
                      {
                        v36 = v8;
                        j_j___libc_free_0_0(v32);
                        v8 = v36;
                      }
                    }
                  }
                  *v27 = -16;
                  v29 = *(_DWORD *)(v22 + 48);
                  ++*(_DWORD *)(v22 + 52);
                  *(_DWORD *)(v22 + 48) = (2 * (v29 >> 1) - 2) | v29 & 1;
                }
                else
                {
                  while ( v28 != -8 )
                  {
                    v26 = v24 & (v25 + v26);
                    v27 = (__int64 *)(v23 + 48LL * v26);
                    v28 = *v27;
                    if ( a2 == *v27 )
                      goto LABEL_23;
                    ++v25;
                  }
                }
LABEL_25:
                v21 += 2;
                if ( v21 == v20 )
                  return;
                while ( *v21 == -8 || *v21 == -16 )
                {
                  v21 += 2;
                  if ( v20 == v21 )
                    return;
                }
              }
            }
          }
        }
      }
      else
      {
        v33 = 1;
        while ( v11 != -8 )
        {
          v34 = v33 + 1;
          v9 = (v6 - 1) & (v33 + v9);
          v10 = (__int64 *)(v7 + 8LL * v9);
          v11 = *v10;
          if ( a2 == *v10 )
            goto LABEL_4;
          v33 = v34;
        }
      }
    }
  }
}

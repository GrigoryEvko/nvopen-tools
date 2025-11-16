// Function: sub_9BA4A0
// Address: 0x9ba4a0
//
void __fastcall sub_9BA4A0(__int64 a1)
{
  int **v2; // r13
  int **v3; // r14
  int *v4; // r12
  int v5; // eax
  __int64 v6; // rsi
  __int64 v7; // rdi
  int v8; // ecx
  unsigned int v9; // r8d
  int *v10; // rdx
  int v11; // r15d
  int v12; // ecx
  int v13; // eax
  unsigned int v14; // r8d
  int *v15; // rdx
  int v16; // r11d
  __int64 v17; // rdx
  int v18; // r9d
  __int64 v19; // r10
  int v20; // r9d
  unsigned int v21; // r8d
  __int64 *v22; // rax
  __int64 v23; // r15
  int v24; // edx
  int v25; // r10d
  int v26; // eax
  int v27; // r11d
  int **v28; // r14
  int *v29; // r12
  int v30; // r9d
  int v31; // eax
  __int64 v32; // rsi
  __int64 v33; // rdi
  int v34; // ecx
  unsigned int v35; // r8d
  int *v36; // rdx
  int v37; // r15d
  int v38; // ecx
  int v39; // eax
  unsigned int v40; // r8d
  int *v41; // rdx
  int v42; // r11d
  __int64 v43; // rdx
  int v44; // r9d
  __int64 v45; // r10
  int v46; // r9d
  unsigned int v47; // r8d
  __int64 *v48; // rax
  __int64 v49; // r15
  int v50; // edx
  int v51; // r10d
  int v52; // eax
  int v53; // r11d
  int v54; // edx
  int v55; // r11d
  int v56; // edx
  int v57; // r11d

  if ( *(_BYTE *)(a1 + 40) )
  {
    v2 = *(int ***)(a1 + 88);
    if ( !*(_BYTE *)(a1 + 108) )
    {
      v3 = &v2[*(unsigned int *)(a1 + 96)];
      if ( v3 == v2 )
      {
LABEL_21:
        *(_BYTE *)(a1 + 40) = 0;
        return;
      }
      while ( 1 )
      {
        v4 = *v2;
        if ( (unsigned __int64)*v2 + 2 <= 1 )
          goto LABEL_5;
        v5 = v4[10];
        v6 = (unsigned int)v4[8];
        v7 = *((_QWORD *)v4 + 2);
        v8 = v5 + *v4 - 1;
        if ( !(_DWORD)v6 )
          goto LABEL_10;
        v9 = (v6 - 1) & (37 * v8);
        v10 = (int *)(v7 + 16LL * v9);
        v11 = *v10;
        if ( v8 != *v10 )
        {
          v54 = 1;
          while ( v11 != 0x7FFFFFFF )
          {
            v55 = v54 + 1;
            v9 = (v6 - 1) & (v54 + v9);
            v10 = (int *)(v7 + 16LL * v9);
            v11 = *v10;
            if ( v8 == *v10 )
              goto LABEL_9;
            v54 = v55;
          }
          goto LABEL_10;
        }
LABEL_9:
        if ( *((_QWORD *)v10 + 1) )
        {
LABEL_5:
          if ( ++v2 == v3 )
            goto LABEL_21;
        }
        else
        {
LABEL_10:
          v12 = 0;
          if ( *v4 )
          {
            while ( 1 )
            {
              v13 = v12 + v5;
              if ( (_DWORD)v6 )
              {
                v14 = (v6 - 1) & (37 * v13);
                v15 = (int *)(v7 + 16LL * v14);
                v16 = *v15;
                if ( *v15 == v13 )
                {
LABEL_15:
                  v17 = *((_QWORD *)v15 + 1);
                  if ( v17 )
                  {
                    v18 = *(_DWORD *)(a1 + 72);
                    v19 = *(_QWORD *)(a1 + 56);
                    if ( v18 )
                    {
                      v20 = v18 - 1;
                      v21 = v20 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
                      v22 = (__int64 *)(v19 + 16LL * v21);
                      v23 = *v22;
                      if ( v17 == *v22 )
                      {
LABEL_18:
                        *v22 = -8192;
                        --*(_DWORD *)(a1 + 64);
                        ++*(_DWORD *)(a1 + 68);
                        v6 = (unsigned int)v4[8];
                        v7 = *((_QWORD *)v4 + 2);
                      }
                      else
                      {
                        v26 = 1;
                        while ( v23 != -4096 )
                        {
                          v27 = v26 + 1;
                          v21 = v20 & (v26 + v21);
                          v22 = (__int64 *)(v19 + 16LL * v21);
                          v23 = *v22;
                          if ( v17 == *v22 )
                            goto LABEL_18;
                          v26 = v27;
                        }
                      }
                    }
                  }
                }
                else
                {
                  v24 = 1;
                  while ( v16 != 0x7FFFFFFF )
                  {
                    v25 = v24 + 1;
                    v14 = (v6 - 1) & (v24 + v14);
                    v15 = (int *)(v7 + 16LL * v14);
                    v16 = *v15;
                    if ( v13 == *v15 )
                      goto LABEL_15;
                    v24 = v25;
                  }
                }
              }
              if ( *v4 <= (unsigned int)++v12 )
                break;
              v5 = v4[10];
            }
          }
          ++v2;
          sub_C7D6A0(v7, 16 * v6, 8);
          j_j___libc_free_0(v4, 56);
          *(v2 - 1) = (int *)-2LL;
          ++*(_DWORD *)(a1 + 104);
          ++*(_QWORD *)(a1 + 80);
          if ( v2 == v3 )
            goto LABEL_21;
        }
      }
    }
    v28 = &v2[*(unsigned int *)(a1 + 100)];
    if ( v28 == v2 )
      goto LABEL_21;
    while ( 1 )
    {
      v29 = *v2;
      v30 = **v2;
      v31 = (*v2)[10];
      v32 = (unsigned int)(*v2)[8];
      v33 = *((_QWORD *)*v2 + 2);
      v34 = v30 + v31 - 1;
      if ( !(_DWORD)v32 )
        goto LABEL_37;
      v35 = (v32 - 1) & (37 * v34);
      v36 = (int *)(v33 + 16LL * v35);
      v37 = *v36;
      if ( v34 != *v36 )
        break;
LABEL_36:
      if ( !*((_QWORD *)v36 + 1) )
        goto LABEL_37;
      ++v2;
LABEL_33:
      if ( v2 == v28 )
        goto LABEL_21;
    }
    v56 = 1;
    while ( v37 != 0x7FFFFFFF )
    {
      v57 = v56 + 1;
      v35 = (v32 - 1) & (v56 + v35);
      v36 = (int *)(v33 + 16LL * v35);
      v37 = *v36;
      if ( v34 == *v36 )
        goto LABEL_36;
      v56 = v57;
    }
LABEL_37:
    v38 = 0;
    if ( v30 )
    {
      while ( 1 )
      {
        v39 = v38 + v31;
        if ( (_DWORD)v32 )
        {
          v40 = (v32 - 1) & (37 * v39);
          v41 = (int *)(v33 + 16LL * v40);
          v42 = *v41;
          if ( *v41 == v39 )
          {
LABEL_42:
            v43 = *((_QWORD *)v41 + 1);
            if ( v43 )
            {
              v44 = *(_DWORD *)(a1 + 72);
              v45 = *(_QWORD *)(a1 + 56);
              if ( v44 )
              {
                v46 = v44 - 1;
                v47 = v46 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                v48 = (__int64 *)(v45 + 16LL * v47);
                v49 = *v48;
                if ( v43 == *v48 )
                {
LABEL_45:
                  *v48 = -8192;
                  --*(_DWORD *)(a1 + 64);
                  ++*(_DWORD *)(a1 + 68);
                  v32 = (unsigned int)v29[8];
                  v33 = *((_QWORD *)v29 + 2);
                }
                else
                {
                  v52 = 1;
                  while ( v49 != -4096 )
                  {
                    v53 = v52 + 1;
                    v47 = v46 & (v52 + v47);
                    v48 = (__int64 *)(v45 + 16LL * v47);
                    v49 = *v48;
                    if ( v43 == *v48 )
                      goto LABEL_45;
                    v52 = v53;
                  }
                }
              }
            }
          }
          else
          {
            v50 = 1;
            while ( v42 != 0x7FFFFFFF )
            {
              v51 = v50 + 1;
              v40 = (v32 - 1) & (v50 + v40);
              v41 = (int *)(v33 + 16LL * v40);
              v42 = *v41;
              if ( v39 == *v41 )
                goto LABEL_42;
              v50 = v51;
            }
          }
        }
        if ( *v29 <= (unsigned int)++v38 )
          break;
        v31 = v29[10];
      }
    }
    --v28;
    sub_C7D6A0(v33, 16 * v32, 8);
    j_j___libc_free_0(v29, 56);
    *v2 = *v28;
    --*(_DWORD *)(a1 + 100);
    ++*(_QWORD *)(a1 + 80);
    goto LABEL_33;
  }
}

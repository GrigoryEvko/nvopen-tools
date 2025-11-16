// Function: sub_38B8D50
// Address: 0x38b8d50
//
void __fastcall sub_38B8D50(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // rax
  int v7; // esi
  __int64 v8; // rcx
  unsigned int v9; // r8d
  __int64 *v10; // rdx
  __int64 v11; // r9
  __int64 *v12; // rdi
  __int64 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r8
  _BYTE *v16; // rax
  __int64 v17; // r8
  int v18; // eax
  __int64 *v19; // rsi
  __int64 v20; // rcx
  _QWORD *v21; // r8
  __int64 v22; // rdi
  __int64 *v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned int v28; // r14d
  int v29; // ecx
  unsigned int v30; // edi
  __int64 *v31; // rsi
  __int64 v32; // r9
  __int64 *v33; // r8
  __int64 *v34; // rax
  __int64 v35; // rax
  __int64 v36; // r8
  _BYTE *v37; // rax
  __int64 v38; // r8
  int v39; // eax
  unsigned int v40; // eax
  __int64 *v41; // r14
  __int64 v42; // rsi
  unsigned __int64 v43; // r15
  unsigned __int64 v44; // rdi
  unsigned int v45; // r12d
  int v46; // edx
  __int64 *v47; // r14
  __int64 v48; // rax
  unsigned __int64 v49; // r12
  unsigned __int64 v50; // rdi
  __int64 v51; // r10
  unsigned int v52; // r11d
  int v53; // r8d
  int v54; // r15d
  __int64 v55; // r10
  int v56; // r14d
  int v57; // edi
  int v58; // r11d
  __int64 v59; // rdi
  int v60; // esi
  int v61; // r8d
  int v62; // edx
  int v63; // edi
  int v64; // r14d
  int v65; // edi
  __int64 v66[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a1 + 288);
  if ( !v4 )
    goto LABEL_3;
  if ( *(_BYTE *)(a1 + 440) )
    goto LABEL_3;
  v26 = *(unsigned int *)(v4 + 48);
  if ( !(_DWORD)v26 )
    goto LABEL_3;
  v27 = *(_QWORD *)(v4 + 32);
  v28 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v29 = v26 - 1;
  v30 = (v26 - 1) & v28;
  v31 = (__int64 *)(v27 + 16LL * v30);
  v32 = *v31;
  v33 = v31;
  if ( a2 != *v31 )
  {
    v51 = *v31;
    v52 = (v26 - 1) & v28;
    v53 = 1;
    while ( v51 != -8 )
    {
      v54 = v53 + 1;
      v52 = v29 & (v53 + v52);
      v33 = (__int64 *)(v27 + 16LL * v52);
      v51 = *v33;
      if ( a2 == *v33 )
        goto LABEL_25;
      v53 = v54;
    }
    goto LABEL_3;
  }
LABEL_25:
  v34 = (__int64 *)(v27 + 16 * v26);
  if ( v34 == v33 || !v33[1] )
    goto LABEL_3;
  if ( a2 != v32 )
  {
    v60 = 1;
    while ( v32 != -8 )
    {
      v61 = v60 + 1;
      v30 = v29 & (v60 + v30);
      v31 = (__int64 *)(v27 + 16LL * v30);
      v32 = *v31;
      if ( a2 == *v31 )
        goto LABEL_28;
      v60 = v61;
    }
LABEL_83:
    v66[0] = 0;
    *(_BYTE *)(v4 + 72) = 0;
    BUG();
  }
LABEL_28:
  if ( v34 == v31 )
    goto LABEL_83;
  v66[0] = v31[1];
  v35 = v66[0];
  *(_BYTE *)(v4 + 72) = 0;
  v36 = *(_QWORD *)(v35 + 8);
  if ( v36 )
  {
    v37 = sub_38B8520(*(_QWORD **)(v36 + 24), *(_QWORD *)(v36 + 32), v66);
    sub_15CDF70(v38 + 24, v37);
    v39 = *(_DWORD *)(v4 + 48);
    v27 = *(_QWORD *)(v4 + 32);
    if ( !v39 )
      goto LABEL_3;
    v29 = v39 - 1;
  }
  v40 = v29 & v28;
  v41 = (__int64 *)(v27 + 16LL * (v29 & v28));
  v42 = *v41;
  if ( a2 == *v41 )
  {
LABEL_33:
    v43 = v41[1];
    if ( v43 )
    {
      v44 = *(_QWORD *)(v43 + 24);
      if ( v44 )
        j_j___libc_free_0(v44);
      j_j___libc_free_0(v43);
    }
    *v41 = -16;
    --*(_DWORD *)(v4 + 40);
    ++*(_DWORD *)(v4 + 44);
  }
  else
  {
    v64 = 1;
    while ( v42 != -8 )
    {
      v65 = v64 + 1;
      v40 = v29 & (v40 + v64);
      v41 = (__int64 *)(v27 + 16LL * v40);
      v42 = *v41;
      if ( a2 == *v41 )
        goto LABEL_33;
      v64 = v65;
    }
  }
LABEL_3:
  v5 = *(_QWORD *)(a1 + 296);
  if ( v5 )
  {
    if ( !*(_BYTE *)(a1 + 441) )
    {
      v6 = *(unsigned int *)(v5 + 72);
      if ( (_DWORD)v6 )
      {
        v7 = v6 - 1;
        v8 = *(_QWORD *)(v5 + 56);
        v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        v12 = v10;
        if ( a2 == *v10 )
        {
LABEL_8:
          v13 = (__int64 *)(v8 + 16 * v6);
          if ( v13 != v12 && v12[1] )
          {
            if ( a2 == v11 )
            {
LABEL_11:
              if ( v10 != v13 )
              {
                v66[0] = v10[1];
                v14 = v66[0];
                *(_BYTE *)(v5 + 96) = 0;
                v15 = *(_QWORD *)(v14 + 8);
                if ( v15 )
                {
                  v16 = sub_38B8520(*(_QWORD **)(v15 + 24), *(_QWORD *)(v15 + 32), v66);
                  sub_15CDF70(v17 + 24, v16);
                  v18 = *(_DWORD *)(v5 + 72);
                  v8 = *(_QWORD *)(v5 + 56);
                  if ( !v18 )
                    goto LABEL_14;
                  v7 = v18 - 1;
                }
                v45 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
                v46 = 1;
                v47 = (__int64 *)(v8 + 16LL * v45);
                v48 = *v47;
                if ( a2 == *v47 )
                {
LABEL_40:
                  v49 = v47[1];
                  if ( v49 )
                  {
                    v50 = *(_QWORD *)(v49 + 24);
                    if ( v50 )
                      j_j___libc_free_0(v50);
                    j_j___libc_free_0(v49);
                  }
                  *v47 = -16;
                  --*(_DWORD *)(v5 + 64);
                  ++*(_DWORD *)(v5 + 68);
                }
                else
                {
                  while ( v48 != -8 )
                  {
                    v45 = v7 & (v46 + v45);
                    v47 = (__int64 *)(v8 + 16LL * v45);
                    v48 = *v47;
                    if ( a2 == *v47 )
                      goto LABEL_40;
                    ++v46;
                  }
                }
LABEL_14:
                v19 = *(__int64 **)v5;
                v20 = 8LL * *(unsigned int *)(v5 + 8);
                v21 = (_QWORD *)(*(_QWORD *)v5 + v20);
                v22 = v20 >> 3;
                if ( v20 >> 5 )
                {
                  v23 = *(__int64 **)v5;
                  while ( a2 != *v23 )
                  {
                    if ( a2 == v23[1] )
                    {
                      ++v23;
                      goto LABEL_21;
                    }
                    if ( a2 == v23[2] )
                    {
                      v23 += 2;
                      goto LABEL_21;
                    }
                    if ( a2 == v23[3] )
                    {
                      v23 += 3;
                      goto LABEL_21;
                    }
                    v23 += 4;
                    if ( v23 == &v19[4 * (v20 >> 5)] )
                    {
                      v22 = v21 - v23;
                      goto LABEL_52;
                    }
                  }
                  goto LABEL_21;
                }
                v23 = *(__int64 **)v5;
LABEL_52:
                if ( v22 != 2 )
                {
                  if ( v22 != 3 )
                  {
                    if ( v22 != 1 )
                      return;
LABEL_55:
                    if ( a2 != *v23 )
                      return;
                    goto LABEL_21;
                  }
                  if ( a2 == *v23 )
                  {
LABEL_21:
                    if ( v21 != v23 )
                    {
                      v24 = &v19[(unsigned __int64)v20 / 8 - 1];
                      v25 = *v23;
                      *v23 = *v24;
                      *v24 = v25;
                      --*(_DWORD *)(v5 + 8);
                    }
                    return;
                  }
                  ++v23;
                }
                if ( a2 != *v23 )
                {
                  ++v23;
                  goto LABEL_55;
                }
                goto LABEL_21;
              }
            }
            else
            {
              v62 = 1;
              while ( v11 != -8 )
              {
                v63 = v62 + 1;
                v9 = v7 & (v62 + v9);
                v10 = (__int64 *)(v8 + 16LL * v9);
                v11 = *v10;
                if ( a2 == *v10 )
                  goto LABEL_11;
                v62 = v63;
              }
            }
            v66[0] = 0;
            *(_BYTE *)(v5 + 96) = 0;
            BUG();
          }
        }
        else
        {
          v55 = *v10;
          v56 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v57 = 1;
          while ( v55 != -8 )
          {
            v58 = v57 + 1;
            v59 = v7 & (unsigned int)(v56 + v57);
            v56 = v59;
            v12 = (__int64 *)(v8 + 16 * v59);
            v55 = *v12;
            if ( a2 == *v12 )
              goto LABEL_8;
            v57 = v58;
          }
        }
      }
    }
  }
}

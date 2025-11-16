// Function: sub_1412F90
// Address: 0x1412f90
//
void __fastcall sub_1412F90(__int64 a1, unsigned __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // edx
  __int64 *v7; // r13
  int v8; // ecx
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rbx
  unsigned __int64 v13; // rsi
  __int64 v14; // rax
  unsigned __int64 v15; // r12
  __int64 v16; // rcx
  int v17; // r8d
  unsigned int v18; // edx
  __int64 *v19; // r13
  __int64 v20; // rsi
  __int64 v21; // rsi
  unsigned __int64 v22; // rax
  unsigned int v23; // edx
  int v24; // r9d
  unsigned int v25; // ecx
  __int64 *v26; // rbx
  __int64 v27; // r8
  _QWORD *v28; // rax
  __int64 v29; // rax
  int v30; // eax
  __int64 v31; // rax
  __int64 v32; // rcx
  int v33; // r8d
  unsigned int v34; // edx
  __int64 *v35; // rbx
  __int64 v36; // rsi
  unsigned __int64 v37; // rdi
  __int64 v38; // r8
  __int64 v39; // rax
  _QWORD *v40; // rcx
  _QWORD *v41; // rax
  __int64 v42; // rdx
  _QWORD *v43; // rsi
  int v44; // eax
  int v45; // r9d
  __int64 v46; // r8
  unsigned int v47; // edi
  __int64 *v48; // rax
  __int64 v49; // r10
  _QWORD *v50; // rax
  _QWORD *v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // rcx
  int v54; // eax
  int v55; // r11d
  unsigned __int64 v56; // rdi

  if ( !*(_DWORD *)(a1 + 48) )
    goto LABEL_2;
  v14 = *(unsigned int *)(a1 + 56);
  v15 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_DWORD)v14 )
  {
    v16 = *(_QWORD *)(a1 + 40);
    v17 = 1;
    v18 = (v14 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
    v19 = (__int64 *)(v16 + 32LL * v18);
    v20 = *v19;
    if ( v15 != *v19 )
    {
      while ( v20 != -8 )
      {
        v18 = (v14 - 1) & (v17 + v18);
        v19 = (__int64 *)(v16 + 32LL * v18);
        v20 = *v19;
        if ( v15 == *v19 )
          goto LABEL_17;
        ++v17;
      }
      goto LABEL_28;
    }
LABEL_17:
    if ( v19 != (__int64 *)(v16 + 32 * v14) )
    {
      v21 = *(_QWORD *)(a1 + 72);
      v22 = v19[2] & 0xFFFFFFFFFFFFFFF8LL;
      if ( ((unsigned __int8)v19[2] & 7u) >= 3 )
        v22 = 0;
      v23 = *(_DWORD *)(a1 + 88);
      if ( v23 )
      {
        v24 = 1;
        v25 = (v23 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v26 = (__int64 *)(v21 + 80LL * v25);
        v27 = *v26;
        if ( v22 == *v26 )
          goto LABEL_22;
        while ( v27 != -8 )
        {
          v25 = (v23 - 1) & (v24 + v25);
          v26 = (__int64 *)(v21 + 80LL * v25);
          v27 = *v26;
          if ( v22 == *v26 )
            goto LABEL_22;
          ++v24;
        }
      }
      v26 = (__int64 *)(v21 + 80LL * v23);
LABEL_22:
      v28 = (_QWORD *)v26[2];
      if ( (_QWORD *)v26[3] == v28 )
      {
        v51 = &v28[*((unsigned int *)v26 + 9)];
        if ( v28 == v51 )
        {
LABEL_72:
          v28 = v51;
        }
        else
        {
          while ( v15 != *v28 )
          {
            if ( v51 == ++v28 )
              goto LABEL_72;
          }
        }
      }
      else
      {
        v28 = (_QWORD *)sub_16CC9F0(v26 + 1, v15);
        if ( v15 == *v28 )
        {
          v52 = v26[3];
          if ( v52 == v26[2] )
            v53 = *((unsigned int *)v26 + 9);
          else
            v53 = *((unsigned int *)v26 + 8);
          v51 = (_QWORD *)(v52 + 8 * v53);
        }
        else
        {
          v29 = v26[3];
          if ( v29 != v26[2] )
            goto LABEL_25;
          v28 = (_QWORD *)(v29 + 8LL * *((unsigned int *)v26 + 9));
          v51 = v28;
        }
      }
      if ( v51 != v28 )
      {
        *v28 = -2;
        v30 = *((_DWORD *)v26 + 10) + 1;
        *((_DWORD *)v26 + 10) = v30;
        goto LABEL_26;
      }
LABEL_25:
      v30 = *((_DWORD *)v26 + 10);
LABEL_26:
      if ( *((_DWORD *)v26 + 9) == v30 )
      {
        v56 = v26[3];
        if ( v56 != v26[2] )
          _libc_free(v56);
        *v26 = -16;
        --*(_DWORD *)(a1 + 80);
        ++*(_DWORD *)(a1 + 84);
      }
      *v19 = -16;
      --*(_DWORD *)(a1 + 48);
      ++*(_DWORD *)(a1 + 52);
    }
  }
LABEL_28:
  if ( *(_BYTE *)(v15 + 16) > 0x17u )
  {
    v31 = *(unsigned int *)(a1 + 88);
    if ( (_DWORD)v31 )
    {
      v32 = *(_QWORD *)(a1 + 72);
      v33 = 1;
      v34 = (v31 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v35 = (__int64 *)(v32 + 80LL * v34);
      v36 = *v35;
      if ( v15 == *v35 )
      {
LABEL_31:
        if ( v35 != (__int64 *)(v32 + 80 * v31) )
        {
          v37 = v35[3];
          v38 = v35[2];
          if ( v37 == v38 )
            v39 = *((unsigned int *)v35 + 9);
          else
            v39 = *((unsigned int *)v35 + 8);
          v40 = (_QWORD *)(v37 + 8 * v39);
          if ( (_QWORD *)v37 != v40 )
          {
            v41 = (_QWORD *)v35[3];
            while ( 1 )
            {
              v42 = *v41;
              v43 = v41;
              if ( *v41 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v40 == ++v41 )
                goto LABEL_38;
            }
            if ( v41 != v40 )
            {
              do
              {
                v44 = *(_DWORD *)(a1 + 56);
                if ( v44 )
                {
                  v45 = v44 - 1;
                  v46 = *(_QWORD *)(a1 + 40);
                  v47 = (v44 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
                  v48 = (__int64 *)(v46 + 32LL * v47);
                  v49 = *v48;
                  if ( *v48 == v42 )
                  {
LABEL_48:
                    *v48 = -16;
                    --*(_DWORD *)(a1 + 48);
                    ++*(_DWORD *)(a1 + 52);
                  }
                  else
                  {
                    v54 = 1;
                    while ( v49 != -8 )
                    {
                      v55 = v54 + 1;
                      v47 = v45 & (v54 + v47);
                      v48 = (__int64 *)(v46 + 32LL * v47);
                      v49 = *v48;
                      if ( *v48 == v42 )
                        goto LABEL_48;
                      v54 = v55;
                    }
                  }
                }
                v50 = v43 + 1;
                if ( v43 + 1 == v40 )
                  break;
                while ( 1 )
                {
                  v42 = *v50;
                  v43 = v50;
                  if ( *v50 < 0xFFFFFFFFFFFFFFFELL )
                    break;
                  if ( v40 == ++v50 )
                    goto LABEL_52;
                }
              }
              while ( v50 != v40 );
LABEL_52:
              v37 = v35[3];
              v38 = v35[2];
            }
          }
LABEL_38:
          if ( v38 != v37 )
            _libc_free(v37);
          *v35 = -16;
          --*(_DWORD *)(a1 + 80);
          ++*(_DWORD *)(a1 + 84);
        }
      }
      else
      {
        while ( v36 != -8 )
        {
          v34 = (v31 - 1) & (v33 + v34);
          v35 = (__int64 *)(v32 + 80LL * v34);
          v36 = *v35;
          if ( v15 == *v35 )
            goto LABEL_31;
          ++v33;
        }
      }
    }
  }
LABEL_2:
  v4 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 104);
    v6 = (v4 - 1) & (a2 ^ (a2 >> 9));
    v7 = (__int64 *)(v5 + 72LL * v6);
    v8 = 1;
    v9 = *v7;
    if ( *v7 == a2 )
    {
LABEL_4:
      if ( v7 != (__int64 *)(v5 + 72 * v4) )
      {
        v10 = v7[2];
        v11 = (v7[3] - v10) >> 4;
        if ( (_DWORD)v11 )
        {
          v12 = 0;
          do
          {
            if ( (*(_DWORD *)(v10 + v12 + 8) & 7u) <= 2 )
            {
              v13 = *(_QWORD *)(v10 + v12 + 8) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v13 )
              {
                sub_1412000(a1 + 128, v13, a2);
                v10 = v7[2];
              }
            }
            v12 += 16;
          }
          while ( v12 != 16LL * (unsigned int)v11 );
        }
        if ( v10 )
          j_j___libc_free_0(v10, v7[4] - v10);
        *v7 = -16;
        --*(_DWORD *)(a1 + 112);
        ++*(_DWORD *)(a1 + 116);
      }
    }
    else
    {
      while ( v9 != -4 )
      {
        v6 = (v4 - 1) & (v8 + v6);
        v7 = (__int64 *)(v5 + 72LL * v6);
        v9 = *v7;
        if ( a2 == *v7 )
          goto LABEL_4;
        ++v8;
      }
    }
  }
}

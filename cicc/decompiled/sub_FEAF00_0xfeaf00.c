// Function: sub_FEAF00
// Address: 0xfeaf00
//
void __fastcall sub_FEAF00(__int64 a1)
{
  int *v1; // r13
  int *v2; // r12
  __int64 v4; // r14
  __int64 v5; // r8
  int v6; // esi
  unsigned int v7; // r9d
  int *v8; // rax
  int v9; // edx
  char v10; // di
  unsigned int v11; // esi
  unsigned int v12; // eax
  int *v13; // r10
  int v14; // edx
  unsigned int v15; // r8d
  int v16; // eax
  int v17; // r15d
  int v18; // eax
  __int64 v19; // rdi
  int v20; // ecx
  unsigned int v21; // edx
  int v22; // eax
  int v23; // ecx
  __int64 v24; // rdi
  int v25; // ecx
  unsigned int v26; // esi
  int v27; // eax
  int v28; // r9d
  int *v29; // r8
  int v30; // r9d

  v1 = *(int **)(a1 + 32);
  v2 = *(int **)(a1 + 24);
  if ( v2 != v1 )
  {
    v4 = a1 + 48;
    while ( 1 )
    {
      v10 = *(_BYTE *)(a1 + 56) & 1;
      if ( v10 )
      {
        v5 = a1 + 64;
        v6 = 3;
      }
      else
      {
        v11 = *(_DWORD *)(a1 + 72);
        v5 = *(_QWORD *)(a1 + 64);
        if ( !v11 )
        {
          v12 = *(_DWORD *)(a1 + 56);
          ++*(_QWORD *)(a1 + 48);
          v13 = 0;
          v14 = (v12 >> 1) + 1;
LABEL_10:
          v15 = 3 * v11;
          goto LABEL_11;
        }
        v6 = v11 - 1;
      }
      v7 = v6 & (37 * *v2);
      v8 = (int *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( *v2 == *v8 )
      {
LABEL_5:
        *((_QWORD *)v8 + 1) = v2;
        v2 += 22;
        if ( v1 == v2 )
          return;
      }
      else
      {
        v17 = 1;
        v13 = 0;
        while ( v9 != -1 )
        {
          if ( !v13 && v9 == -2 )
            v13 = v8;
          v7 = v6 & (v17 + v7);
          v8 = (int *)(v5 + 16LL * v7);
          v9 = *v8;
          if ( *v2 == *v8 )
            goto LABEL_5;
          ++v17;
        }
        v15 = 12;
        v11 = 4;
        if ( !v13 )
          v13 = v8;
        v12 = *(_DWORD *)(a1 + 56);
        ++*(_QWORD *)(a1 + 48);
        v14 = (v12 >> 1) + 1;
        if ( !v10 )
        {
          v11 = *(_DWORD *)(a1 + 72);
          goto LABEL_10;
        }
LABEL_11:
        if ( 4 * v14 >= v15 )
        {
          sub_FE6000(v4, 2 * v11);
          if ( (*(_BYTE *)(a1 + 56) & 1) != 0 )
          {
            v19 = a1 + 64;
            v20 = 3;
          }
          else
          {
            v18 = *(_DWORD *)(a1 + 72);
            v19 = *(_QWORD *)(a1 + 64);
            if ( !v18 )
              goto LABEL_55;
            v20 = v18 - 1;
          }
          v21 = v20 & (37 * *v2);
          v13 = (int *)(v19 + 16LL * v21);
          v22 = *v13;
          if ( *v13 != *v2 )
          {
            v30 = 1;
            v29 = 0;
            while ( v22 != -1 )
            {
              if ( !v29 && v22 == -2 )
                v29 = v13;
              v21 = v20 & (v30 + v21);
              v13 = (int *)(v19 + 16LL * v21);
              v22 = *v13;
              if ( *v2 == *v13 )
                goto LABEL_27;
              ++v30;
            }
LABEL_34:
            if ( v29 )
              v13 = v29;
          }
LABEL_27:
          v12 = *(_DWORD *)(a1 + 56);
          goto LABEL_13;
        }
        if ( v11 - *(_DWORD *)(a1 + 60) - v14 <= v11 >> 3 )
        {
          sub_FE6000(v4, v11);
          if ( (*(_BYTE *)(a1 + 56) & 1) != 0 )
          {
            v24 = a1 + 64;
            v25 = 3;
          }
          else
          {
            v23 = *(_DWORD *)(a1 + 72);
            v24 = *(_QWORD *)(a1 + 64);
            if ( !v23 )
            {
LABEL_55:
              *(_DWORD *)(a1 + 56) = (2 * (*(_DWORD *)(a1 + 56) >> 1) + 2) | *(_DWORD *)(a1 + 56) & 1;
              BUG();
            }
            v25 = v23 - 1;
          }
          v26 = v25 & (37 * *v2);
          v13 = (int *)(v24 + 16LL * v26);
          v27 = *v13;
          if ( *v13 != *v2 )
          {
            v28 = 1;
            v29 = 0;
            while ( v27 != -1 )
            {
              if ( v27 == -2 && !v29 )
                v29 = v13;
              v26 = v25 & (v28 + v26);
              v13 = (int *)(v24 + 16LL * v26);
              v27 = *v13;
              if ( *v2 == *v13 )
                goto LABEL_27;
              ++v28;
            }
            goto LABEL_34;
          }
          goto LABEL_27;
        }
LABEL_13:
        *(_DWORD *)(a1 + 56) = (2 * (v12 >> 1) + 2) | v12 & 1;
        if ( *v13 != -1 )
          --*(_DWORD *)(a1 + 60);
        v16 = *v2;
        *((_QWORD *)v13 + 1) = 0;
        *v13 = v16;
        *((_QWORD *)v13 + 1) = v2;
        v2 += 22;
        if ( v1 == v2 )
          return;
      }
    }
  }
}

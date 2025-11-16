// Function: sub_150AAB0
// Address: 0x150aab0
//
__int64 __fastcall sub_150AAB0(__int64 a1, int a2)
{
  __int64 v2; // r9
  unsigned int v4; // esi
  __int64 v6; // r8
  unsigned int v7; // edi
  int *v8; // rax
  int v9; // ecx
  int v12; // r11d
  int *v13; // rdx
  int v14; // eax
  int v15; // ecx
  int v16; // eax
  int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // eax
  int v20; // edi
  int v21; // r10d
  int *v22; // r9
  int v23; // eax
  int v24; // eax
  __int64 v25; // rdi
  int *v26; // r8
  unsigned int v27; // r13d
  int v28; // r9d
  int v29; // esi
  int *v30; // r14

  v2 = a1 + 448;
  v4 = *(_DWORD *)(a1 + 472);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 448);
    goto LABEL_15;
  }
  v6 = *(_QWORD *)(a1 + 456);
  v7 = (v4 - 1) & (37 * a2);
  v8 = (int *)(v6 + 24LL * v7);
  v9 = *v8;
  if ( *v8 == a2 )
    return *((_QWORD *)v8 + 1);
  v12 = 1;
  v13 = 0;
  while ( 1 )
  {
    if ( v9 == -1 )
    {
      if ( !v13 )
        v13 = v8;
      v14 = *(_DWORD *)(a1 + 464);
      ++*(_QWORD *)(a1 + 448);
      v15 = v14 + 1;
      if ( 4 * (v14 + 1) < 3 * v4 )
      {
        if ( v4 - *(_DWORD *)(a1 + 468) - v15 > v4 >> 3 )
        {
LABEL_11:
          *(_DWORD *)(a1 + 464) = v15;
          if ( *v13 != -1 )
            --*(_DWORD *)(a1 + 468);
          *v13 = a2;
          *((_QWORD *)v13 + 1) = 0;
          *((_QWORD *)v13 + 2) = 0;
          return 0;
        }
        sub_1509F90(v2, v4);
        v23 = *(_DWORD *)(a1 + 472);
        if ( v23 )
        {
          v24 = v23 - 1;
          v25 = *(_QWORD *)(a1 + 456);
          v26 = 0;
          v27 = v24 & (37 * a2);
          v28 = 1;
          v15 = *(_DWORD *)(a1 + 464) + 1;
          v13 = (int *)(v25 + 24LL * v27);
          v29 = *v13;
          if ( *v13 != a2 )
          {
            while ( v29 != -1 )
            {
              if ( !v26 && v29 == -2 )
                v26 = v13;
              v27 = v24 & (v28 + v27);
              v13 = (int *)(v25 + 24LL * v27);
              v29 = *v13;
              if ( *v13 == a2 )
                goto LABEL_11;
              ++v28;
            }
            if ( v26 )
              v13 = v26;
          }
          goto LABEL_11;
        }
LABEL_44:
        ++*(_DWORD *)(a1 + 464);
        BUG();
      }
LABEL_15:
      sub_1509F90(v2, 2 * v4);
      v16 = *(_DWORD *)(a1 + 472);
      if ( v16 )
      {
        v17 = v16 - 1;
        v18 = *(_QWORD *)(a1 + 456);
        v19 = (v16 - 1) & (37 * a2);
        v15 = *(_DWORD *)(a1 + 464) + 1;
        v13 = (int *)(v18 + 24LL * v19);
        v20 = *v13;
        if ( *v13 != a2 )
        {
          v21 = 1;
          v22 = 0;
          while ( v20 != -1 )
          {
            if ( !v22 && v20 == -2 )
              v22 = v13;
            v19 = v17 & (v21 + v19);
            v13 = (int *)(v18 + 24LL * v19);
            v20 = *v13;
            if ( *v13 == a2 )
              goto LABEL_11;
            ++v21;
          }
          if ( v22 )
            v13 = v22;
        }
        goto LABEL_11;
      }
      goto LABEL_44;
    }
    if ( v13 || v9 != -2 )
      v8 = v13;
    v7 = (v4 - 1) & (v12 + v7);
    v30 = (int *)(v6 + 24LL * v7);
    v9 = *v30;
    if ( *v30 == a2 )
      return *((_QWORD *)v30 + 1);
    ++v12;
    v13 = v8;
    v8 = (int *)(v6 + 24LL * v7);
  }
}

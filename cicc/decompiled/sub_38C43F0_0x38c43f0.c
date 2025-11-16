// Function: sub_38C43F0
// Address: 0x38c43f0
//
__int64 __fastcall sub_38C43F0(__int64 a1, int a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  int *v8; // rax
  int v9; // r9d
  _DWORD *v10; // rdx
  __int64 result; // rax
  int v12; // r11d
  int *v13; // r13
  int v14; // eax
  int v15; // edx
  _DWORD *v16; // rdx
  int v17; // eax
  int v18; // ecx
  __int64 v19; // rdi
  unsigned int v20; // eax
  int v21; // esi
  int v22; // r9d
  int *v23; // r8
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  int v27; // r8d
  unsigned int v28; // r14d
  int *v29; // rdi
  int v30; // ecx

  v4 = a1 + 696;
  v5 = *(_DWORD *)(a1 + 720);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 696);
    goto LABEL_16;
  }
  v6 = *(_QWORD *)(a1 + 704);
  v7 = (v5 - 1) & (37 * a2);
  v8 = (int *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( *v8 != a2 )
  {
    v12 = 1;
    v13 = 0;
    while ( v9 != -1 )
    {
      if ( !v13 && v9 == -2 )
        v13 = v8;
      v7 = (v5 - 1) & (v12 + v7);
      v8 = (int *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == a2 )
        goto LABEL_3;
      ++v12;
    }
    if ( !v13 )
      v13 = v8;
    v14 = *(_DWORD *)(a1 + 712);
    ++*(_QWORD *)(a1 + 696);
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 716) - v15 > v5 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 712) = v15;
        if ( *v13 != -1 )
          --*(_DWORD *)(a1 + 716);
        *v13 = a2;
        *((_QWORD *)v13 + 1) = 0;
        goto LABEL_14;
      }
      sub_38C4230(v4, v5);
      v24 = *(_DWORD *)(a1 + 720);
      if ( v24 )
      {
        v25 = v24 - 1;
        v26 = *(_QWORD *)(a1 + 704);
        v27 = 1;
        v28 = v25 & (37 * a2);
        v15 = *(_DWORD *)(a1 + 712) + 1;
        v29 = 0;
        v13 = (int *)(v26 + 16LL * v28);
        v30 = *v13;
        if ( *v13 != a2 )
        {
          while ( v30 != -1 )
          {
            if ( !v29 && v30 == -2 )
              v29 = v13;
            v28 = v25 & (v27 + v28);
            v13 = (int *)(v26 + 16LL * v28);
            v30 = *v13;
            if ( *v13 == a2 )
              goto LABEL_11;
            ++v27;
          }
          if ( v29 )
            v13 = v29;
        }
        goto LABEL_11;
      }
LABEL_45:
      ++*(_DWORD *)(a1 + 712);
      BUG();
    }
LABEL_16:
    sub_38C4230(v4, 2 * v5);
    v17 = *(_DWORD *)(a1 + 720);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 704);
      v20 = (v17 - 1) & (37 * a2);
      v15 = *(_DWORD *)(a1 + 712) + 1;
      v13 = (int *)(v19 + 16LL * v20);
      v21 = *v13;
      if ( *v13 != a2 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -1 )
        {
          if ( !v23 && v21 == -2 )
            v23 = v13;
          v20 = v18 & (v22 + v20);
          v13 = (int *)(v19 + 16LL * v20);
          v21 = *v13;
          if ( *v13 == a2 )
            goto LABEL_11;
          ++v22;
        }
        if ( v23 )
          v13 = v23;
      }
      goto LABEL_11;
    }
    goto LABEL_45;
  }
LABEL_3:
  v10 = (_DWORD *)*((_QWORD *)v8 + 1);
  if ( v10 )
  {
    result = (unsigned int)(*v10 + 1);
    *v10 = result;
    return result;
  }
  v13 = v8;
LABEL_14:
  v16 = (_DWORD *)sub_145CBF0((__int64 *)(a1 + 48), 4, 8);
  *v16 = 0;
  *((_QWORD *)v13 + 1) = v16;
  result = (unsigned int)(*v16 + 1);
  *v16 = result;
  return result;
}

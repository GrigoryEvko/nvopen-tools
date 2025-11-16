// Function: sub_38CFF00
// Address: 0x38cff00
//
bool __fastcall sub_38CFF00(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rbx
  unsigned int v6; // esi
  __int64 v7; // rcx
  unsigned int v8; // edx
  _QWORD *v9; // rax
  __int64 v10; // r11
  __int64 v11; // rax
  __int64 v12; // rbx
  bool result; // al
  int v14; // r15d
  _QWORD *v15; // r9
  int v16; // eax
  int v17; // edx
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rsi
  int v23; // r10d
  _QWORD *v24; // r8
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  int v28; // r8d
  _QWORD *v29; // rdi
  unsigned int v30; // r14d
  __int64 v31; // rcx

  v4 = a1 + 152;
  v5 = *(_QWORD *)(a2 + 24);
  v6 = *(_DWORD *)(a1 + 176);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 152);
    goto LABEL_19;
  }
  v7 = *(_QWORD *)(a1 + 160);
  v8 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v9 = (_QWORD *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( v5 != *v9 )
  {
    v14 = 1;
    v15 = 0;
    while ( v10 != -8 )
    {
      if ( !v15 && v10 == -16 )
        v15 = v9;
      v8 = (v6 - 1) & (v14 + v8);
      v9 = (_QWORD *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( v5 == *v9 )
        goto LABEL_3;
      ++v14;
    }
    if ( !v15 )
      v15 = v9;
    v16 = *(_DWORD *)(a1 + 168);
    ++*(_QWORD *)(a1 + 152);
    v17 = v16 + 1;
    if ( 4 * (v16 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 172) - v17 > v6 >> 3 )
      {
LABEL_14:
        *(_DWORD *)(a1 + 168) = v17;
        if ( *v15 != -8 )
          --*(_DWORD *)(a1 + 172);
        *v15 = v5;
        v15[1] = 0;
        goto LABEL_17;
      }
      sub_38CFAA0(v4, v6);
      v25 = *(_DWORD *)(a1 + 176);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(a1 + 160);
        v28 = 1;
        v29 = 0;
        v30 = v26 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v17 = *(_DWORD *)(a1 + 168) + 1;
        v15 = (_QWORD *)(v27 + 16LL * v30);
        v31 = *v15;
        if ( v5 != *v15 )
        {
          while ( v31 != -8 )
          {
            if ( v31 == -16 && !v29 )
              v29 = v15;
            v30 = v26 & (v28 + v30);
            v15 = (_QWORD *)(v27 + 16LL * v30);
            v31 = *v15;
            if ( v5 == *v15 )
              goto LABEL_14;
            ++v28;
          }
          if ( v29 )
            v15 = v29;
        }
        goto LABEL_14;
      }
LABEL_47:
      ++*(_DWORD *)(a1 + 168);
      BUG();
    }
LABEL_19:
    sub_38CFAA0(v4, 2 * v6);
    v18 = *(_DWORD *)(a1 + 176);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 160);
      v21 = (v18 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v17 = *(_DWORD *)(a1 + 168) + 1;
      v15 = (_QWORD *)(v20 + 16LL * v21);
      v22 = *v15;
      if ( v5 != *v15 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -8 )
        {
          if ( !v24 && v22 == -16 )
            v24 = v15;
          v21 = v19 & (v23 + v21);
          v15 = (_QWORD *)(v20 + 16LL * v21);
          v22 = *v15;
          if ( v5 == *v15 )
            goto LABEL_14;
          ++v23;
        }
        if ( v24 )
          v15 = v24;
      }
      goto LABEL_14;
    }
    goto LABEL_47;
  }
LABEL_3:
  v11 = v9[1];
  if ( v11 )
  {
    v12 = *(_QWORD *)(v11 + 8);
    goto LABEL_6;
  }
LABEL_17:
  v12 = *(_QWORD *)(v5 + 104);
LABEL_6:
  while ( 1 )
  {
    result = sub_38CF4D0(a1, a2);
    if ( result )
      break;
    sub_390D660(a1, v12);
    v12 = *(_QWORD *)(v12 + 8);
  }
  return result;
}

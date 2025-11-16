// Function: sub_38CFC60
// Address: 0x38cfc60
//
char __fastcall sub_38CFC60(__int64 a1, _QWORD *a2)
{
  _QWORD *v4; // rax
  unsigned int v5; // esi
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // r8
  unsigned int v9; // ecx
  __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  int v12; // r11d
  _QWORD *v13; // r10
  int v14; // ecx
  int v15; // ecx
  int v16; // eax
  int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // edx
  __int64 v20; // rdi
  int v21; // r10d
  _QWORD *v22; // r9
  int v23; // eax
  int v24; // edx
  __int64 v25; // rdi
  _QWORD *v26; // r8
  unsigned int v27; // r14d
  int v28; // r9d
  __int64 v29; // rsi

  LOBYTE(v4) = sub_38CF4D0(a1, (__int64)a2);
  if ( !(_BYTE)v4 )
    return (char)v4;
  v5 = *(_DWORD *)(a1 + 176);
  v6 = a2[3];
  v7 = a1 + 152;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 152);
    goto LABEL_18;
  }
  v8 = *(_QWORD *)(a1 + 160);
  v9 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v4 = (_QWORD *)(v8 + 16LL * v9);
  v10 = *v4;
  if ( v6 != *v4 )
  {
    v12 = 1;
    v13 = 0;
    while ( v10 != -8 )
    {
      if ( !v13 && v10 == -16 )
        v13 = v4;
      v9 = (v5 - 1) & (v12 + v9);
      v4 = (_QWORD *)(v8 + 16LL * v9);
      v10 = *v4;
      if ( v6 == *v4 )
        goto LABEL_4;
      ++v12;
    }
    v14 = *(_DWORD *)(a1 + 168);
    if ( v13 )
      v4 = v13;
    ++*(_QWORD *)(a1 + 152);
    v15 = v14 + 1;
    if ( 4 * v15 < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 172) - v15 > v5 >> 3 )
      {
LABEL_14:
        *(_DWORD *)(a1 + 168) = v15;
        if ( *v4 != -8 )
          --*(_DWORD *)(a1 + 172);
        *v4 = v6;
        v4[1] = 0;
        v10 = a2[3];
        goto LABEL_4;
      }
      sub_38CFAA0(v7, v5);
      v23 = *(_DWORD *)(a1 + 176);
      if ( v23 )
      {
        v24 = v23 - 1;
        v25 = *(_QWORD *)(a1 + 160);
        v26 = 0;
        v27 = (v23 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v28 = 1;
        v15 = *(_DWORD *)(a1 + 168) + 1;
        v4 = (_QWORD *)(v25 + 16LL * v27);
        v29 = *v4;
        if ( v6 != *v4 )
        {
          while ( v29 != -8 )
          {
            if ( v29 == -16 && !v26 )
              v26 = v4;
            v27 = v24 & (v28 + v27);
            v4 = (_QWORD *)(v25 + 16LL * v27);
            v29 = *v4;
            if ( v6 == *v4 )
              goto LABEL_14;
            ++v28;
          }
          if ( v26 )
            v4 = v26;
        }
        goto LABEL_14;
      }
LABEL_46:
      ++*(_DWORD *)(a1 + 168);
      BUG();
    }
LABEL_18:
    sub_38CFAA0(v7, 2 * v5);
    v16 = *(_DWORD *)(a1 + 176);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 160);
      v19 = (v16 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v15 = *(_DWORD *)(a1 + 168) + 1;
      v4 = (_QWORD *)(v18 + 16LL * v19);
      v20 = *v4;
      if ( v6 != *v4 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -8 )
        {
          if ( !v22 && v20 == -16 )
            v22 = v4;
          v19 = v17 & (v21 + v19);
          v4 = (_QWORD *)(v18 + 16LL * v19);
          v20 = *v4;
          if ( v6 == *v4 )
            goto LABEL_14;
          ++v21;
        }
        if ( v22 )
          v4 = v22;
      }
      goto LABEL_14;
    }
    goto LABEL_46;
  }
LABEL_4:
  if ( a2 == *(_QWORD **)(v10 + 104) )
    v11 = 0;
  else
    v11 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v4[1] = v11;
  return (char)v4;
}

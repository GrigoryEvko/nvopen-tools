// Function: sub_39BFF80
// Address: 0x39bff80
//
__int64 __fastcall sub_39BFF80(__int64 a1, __int64 a2, char a3)
{
  unsigned int v5; // esi
  unsigned int v6; // r13d
  __int64 v7; // rdi
  int v8; // r15d
  __int64 *v9; // r9
  unsigned int v10; // ecx
  unsigned int *v11; // rax
  __int64 v12; // r11
  int v14; // ecx
  int v15; // eax
  int v16; // esi
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // r8
  int v20; // r11d
  __int64 *v21; // r10
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdi
  __int64 *v25; // r8
  unsigned int v26; // r14d
  int v27; // r10d
  __int64 v28; // rsi
  char v29; // [rsp+Ch] [rbp-34h]
  char v30; // [rsp+Ch] [rbp-34h]

  v5 = *(_DWORD *)(a1 + 24);
  v6 = *(_DWORD *)(a1 + 16);
  *(_BYTE *)(a1 + 32) = 1;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_19;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = 0;
  v10 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (unsigned int *)(v7 + 16LL * v10);
  v12 = *(_QWORD *)v11;
  if ( a2 == *(_QWORD *)v11 )
    return v11[2];
  while ( v12 != -8 )
  {
    if ( !v9 && v12 == -16 )
      v9 = (__int64 *)v11;
    v10 = (v5 - 1) & (v8 + v10);
    v11 = (unsigned int *)(v7 + 16LL * v10);
    v12 = *(_QWORD *)v11;
    if ( a2 == *(_QWORD *)v11 )
      return v11[2];
    ++v8;
  }
  v14 = v6 + 1;
  if ( !v9 )
    v9 = (__int64 *)v11;
  ++*(_QWORD *)a1;
  if ( 4 * v14 >= 3 * v5 )
  {
LABEL_19:
    v29 = a3;
    sub_39BFDC0(a1, 2 * v5);
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      a3 = v29;
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 16) + 1;
      v9 = (__int64 *)(v17 + 16LL * v18);
      v19 = *v9;
      if ( a2 != *v9 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -8 )
        {
          if ( !v21 && v19 == -16 )
            v21 = v9;
          v18 = v16 & (v20 + v18);
          v9 = (__int64 *)(v17 + 16LL * v18);
          v19 = *v9;
          if ( a2 == *v9 )
            goto LABEL_15;
          ++v20;
        }
        if ( v21 )
          v9 = v21;
      }
      goto LABEL_15;
    }
    goto LABEL_42;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v14 <= v5 >> 3 )
  {
    v30 = a3;
    sub_39BFDC0(a1, v5);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 8);
      v25 = 0;
      v26 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      a3 = v30;
      v27 = 1;
      v14 = *(_DWORD *)(a1 + 16) + 1;
      v9 = (__int64 *)(v24 + 16LL * v26);
      v28 = *v9;
      if ( a2 != *v9 )
      {
        while ( v28 != -8 )
        {
          if ( !v25 && v28 == -16 )
            v25 = v9;
          v26 = v23 & (v27 + v26);
          v9 = (__int64 *)(v24 + 16LL * v26);
          v28 = *v9;
          if ( a2 == *v9 )
            goto LABEL_15;
          ++v27;
        }
        if ( v25 )
          v9 = v25;
      }
      goto LABEL_15;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v14;
  if ( *v9 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v9 = a2;
  *((_DWORD *)v9 + 2) = v6;
  *((_BYTE *)v9 + 12) = a3;
  return v6;
}

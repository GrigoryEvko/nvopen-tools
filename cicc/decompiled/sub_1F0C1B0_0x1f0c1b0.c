// Function: sub_1F0C1B0
// Address: 0x1f0c1b0
//
__int64 __fastcall sub_1F0C1B0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 *v4; // r14
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  int v13; // r8d
  unsigned int v14; // edx
  __int64 *v15; // rcx
  __int64 v16; // r9
  __int64 *v17; // rdx
  __int64 *v18; // rax
  unsigned int v19; // r9d
  __int64 *v20; // rcx
  __int64 v21; // r11
  __int64 *v22; // rdx
  __int64 *v23; // rcx
  int v25; // ecx
  int v26; // ecx
  int v27; // r10d
  int v28; // r10d

  if ( a2 == a3 )
    return 0;
  v4 = a2;
  v7 = a1;
  do
  {
    v8 = *v4;
    sub_1E06620(a4);
    v9 = *(_QWORD *)(*(_QWORD *)(v7 + 56) + 328LL);
    if ( v9 == v7 || v8 == v9 )
    {
      v7 = *(_QWORD *)(*(_QWORD *)(v7 + 56) + 328LL);
      goto LABEL_20;
    }
    v10 = *(_QWORD *)(a4 + 1312);
    v11 = *(_QWORD *)(v10 + 32);
    v12 = *(unsigned int *)(v10 + 48);
    if ( !(_DWORD)v12 )
      return 0;
    v13 = v12 - 1;
    v14 = (v12 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( *v15 != v7 )
    {
      v25 = 1;
      while ( v16 != -8 )
      {
        v28 = v25 + 1;
        v14 = v13 & (v25 + v14);
        v15 = (__int64 *)(v11 + 16LL * v14);
        v16 = *v15;
        if ( *v15 == v7 )
          goto LABEL_7;
        v25 = v28;
      }
      v17 = (__int64 *)(v11 + 16 * v12);
LABEL_28:
      v18 = 0;
      goto LABEL_9;
    }
LABEL_7:
    v17 = (__int64 *)(v11 + 16 * v12);
    if ( v17 == v15 )
      goto LABEL_28;
    v18 = (__int64 *)v15[1];
LABEL_9:
    v19 = v13 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v20 = (__int64 *)(v11 + 16LL * v19);
    v21 = *v20;
    if ( *v20 != v8 )
    {
      v26 = 1;
      while ( v21 != -8 )
      {
        v27 = v26 + 1;
        v19 = v13 & (v26 + v19);
        v20 = (__int64 *)(v11 + 16LL * v19);
        v21 = *v20;
        if ( v8 == *v20 )
          goto LABEL_10;
        v26 = v27;
      }
      return 0;
    }
LABEL_10:
    if ( v17 == v20 )
      return 0;
    v22 = (__int64 *)v20[1];
    if ( !v18 || !v22 )
      return 0;
    while ( v18 != v22 )
    {
      if ( *((_DWORD *)v18 + 4) < *((_DWORD *)v22 + 4) )
      {
        v23 = v18;
        v18 = v22;
        v22 = v23;
      }
      v18 = (__int64 *)v18[1];
      if ( !v18 )
        return 0;
    }
    v7 = *v18;
LABEL_20:
    if ( !v7 )
      return 0;
    ++v4;
  }
  while ( a3 != v4 );
  if ( a1 == v7 )
    return 0;
  return v7;
}

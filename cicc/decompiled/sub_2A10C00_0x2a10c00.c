// Function: sub_2A10C00
// Address: 0x2a10c00
//
__int64 __fastcall sub_2A10C00(_QWORD *a1, __int64 **a2, __int64 a3)
{
  __int64 *v3; // r14
  __int64 *v4; // r15
  int v5; // r10d
  __int64 v6; // rbx
  int v7; // r11d
  __int64 v8; // r13
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rcx
  _QWORD *v12; // rax
  __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r9
  __int64 v18; // rsi
  unsigned int v19; // ecx
  __int64 *v20; // rdx
  __int64 v21; // r8
  _QWORD *v22; // rcx
  _QWORD *v23; // rdx
  int v25; // edx
  int v26; // eax
  int v27; // esi
  int v28; // [rsp+0h] [rbp-2Ch]

  v3 = *a2;
  v4 = a2[1];
  if ( *a2 == v4 )
    return 0;
  v5 = *(_DWORD *)(a3 + 24);
  v6 = *(_QWORD *)(a3 + 8);
  v7 = v5 - 1;
  while ( 1 )
  {
    v8 = *v3;
    if ( !v5 )
      goto LABEL_33;
    v9 = v7 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v8 != *v10 )
    {
      v26 = 1;
      while ( v11 != -4096 )
      {
        v27 = v26 + 1;
        v9 = v7 & (v26 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( v8 == *v10 )
          goto LABEL_5;
        v26 = v27;
      }
LABEL_33:
      v12 = 0;
      goto LABEL_6;
    }
LABEL_5:
    v12 = (_QWORD *)v10[1];
LABEL_6:
    if ( a1 != v12 )
    {
      v13 = *(_QWORD *)(v8 + 56);
      v14 = v8 + 48;
      if ( v13 != v14 )
        break;
    }
LABEL_26:
    if ( v4 == ++v3 )
      return 0;
  }
  while ( 1 )
  {
    if ( !v13 )
      BUG();
    v15 = 32LL * (*(_DWORD *)(v13 - 20) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v13 - 17) & 0x40) != 0 )
    {
      v16 = *(_QWORD *)(v13 - 32);
      v17 = v16 + v15;
    }
    else
    {
      v17 = v13 - 24;
      v16 = v13 - 24 - v15;
    }
    if ( v16 != v17 )
      break;
LABEL_25:
    v13 = *(_QWORD *)(v13 + 8);
    if ( v14 == v13 )
      goto LABEL_26;
  }
  while ( 2 )
  {
    if ( **(_BYTE **)v16 <= 0x1Cu || (v18 = *(_QWORD *)(*(_QWORD *)v16 + 40LL), !v5) )
    {
LABEL_24:
      v16 += 32;
      if ( v17 == v16 )
        goto LABEL_25;
      continue;
    }
    break;
  }
  v19 = v7 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
  v20 = (__int64 *)(v6 + 16LL * v19);
  v21 = *v20;
  if ( v18 != *v20 )
  {
    v25 = 1;
    while ( v21 != -4096 )
    {
      v19 = v7 & (v25 + v19);
      v28 = v25 + 1;
      v20 = (__int64 *)(v6 + 16LL * v19);
      v21 = *v20;
      if ( v18 == *v20 )
        goto LABEL_15;
      v25 = v28;
    }
    goto LABEL_24;
  }
LABEL_15:
  v22 = (_QWORD *)v20[1];
  if ( !v22 )
    goto LABEL_24;
  if ( v22 == a1 )
    return 1;
  if ( !a1 )
    goto LABEL_24;
  v23 = a1;
  while ( 1 )
  {
    v23 = (_QWORD *)*v23;
    if ( v22 == v23 )
      return 1;
    if ( !v23 )
      goto LABEL_24;
  }
}

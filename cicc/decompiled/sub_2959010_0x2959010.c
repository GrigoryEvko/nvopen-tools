// Function: sub_2959010
// Address: 0x2959010
//
bool __fastcall sub_2959010(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r8
  int v6; // esi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r11
  unsigned int v10; // edx
  _QWORD *j; // rax
  unsigned int v12; // edi
  __int64 *v13; // rax
  __int64 v14; // r9
  unsigned int v15; // ecx
  _QWORD *k; // rax
  int v18; // esi
  int i; // eax
  int v20; // ebx
  int v21; // eax
  int v22; // r10d

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v5 = a1 + 16;
    v6 = 15;
  }
  else
  {
    v18 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    if ( !v18 )
      BUG();
    v6 = v18 - 1;
  }
  v7 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    for ( i = 1; ; i = v20 )
    {
      if ( v9 == -4096 )
        BUG();
      v20 = i + 1;
      v7 = v6 & (i + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        break;
    }
  }
  v10 = 1;
  for ( j = *(_QWORD **)v8[1]; j; ++v10 )
    j = (_QWORD *)*j;
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 && !*(_DWORD *)(a1 + 24) )
    goto LABEL_25;
  v12 = v6 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v13 = (__int64 *)(v5 + 16LL * v12);
  v14 = *v13;
  if ( a3 != *v13 )
  {
    v21 = 1;
    while ( v14 != -4096 )
    {
      v22 = v21 + 1;
      v12 = v6 & (v21 + v12);
      v13 = (__int64 *)(v5 + 16LL * v12);
      v14 = *v13;
      if ( a3 == *v13 )
        goto LABEL_8;
      v21 = v22;
    }
LABEL_25:
    BUG();
  }
LABEL_8:
  v15 = 1;
  for ( k = *(_QWORD **)v13[1]; k; ++v15 )
    k = (_QWORD *)*k;
  return v15 > v10;
}

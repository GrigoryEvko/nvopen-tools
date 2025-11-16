// Function: sub_1A4F560
// Address: 0x1a4f560
//
bool __fastcall sub_1A4F560(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r10
  char v5; // r8
  __int64 v6; // rdi
  int v7; // esi
  unsigned int v8; // r9d
  __int64 *v9; // rax
  __int64 v10; // r11
  unsigned int v11; // ecx
  _QWORD *j; // rax
  unsigned int v13; // r8d
  __int64 *v14; // rax
  __int64 v15; // r9
  unsigned int v16; // edx
  _QWORD *k; // rax
  int v19; // esi
  int v20; // eax
  int v21; // r10d
  int i; // eax
  int v23; // ebx

  v3 = *a1;
  v5 = *(_BYTE *)(*a1 + 8) & 1;
  if ( v5 )
  {
    v6 = v3 + 16;
    v7 = 15;
  }
  else
  {
    v19 = *(_DWORD *)(v3 + 24);
    v6 = *(_QWORD *)(v3 + 16);
    if ( !v19 )
      BUG();
    v7 = v19 - 1;
  }
  v8 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v6 + 16LL * v8);
  v10 = *v9;
  if ( *v9 != a2 )
  {
    for ( i = 1; ; i = v23 )
    {
      if ( v10 == -8 )
        BUG();
      v23 = i + 1;
      v8 = v7 & (i + v8);
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        break;
    }
  }
  v11 = 1;
  for ( j = *(_QWORD **)v9[1]; j; ++v11 )
    j = (_QWORD *)*j;
  if ( !v5 && !*(_DWORD *)(v3 + 24) )
    goto LABEL_25;
  v13 = v7 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v14 = (__int64 *)(v6 + 16LL * v13);
  v15 = *v14;
  if ( a3 != *v14 )
  {
    v20 = 1;
    while ( v15 != -8 )
    {
      v21 = v20 + 1;
      v13 = v7 & (v20 + v13);
      v14 = (__int64 *)(v6 + 16LL * v13);
      v15 = *v14;
      if ( a3 == *v14 )
        goto LABEL_8;
      v20 = v21;
    }
LABEL_25:
    BUG();
  }
LABEL_8:
  v16 = 1;
  for ( k = *(_QWORD **)v14[1]; k; ++v16 )
    k = (_QWORD *)*k;
  return v11 < v16;
}

// Function: sub_34BC290
// Address: 0x34bc290
//
char __fastcall sub_34BC290(int **a1, int *a2, int *a3)
{
  unsigned int v4; // r11d
  int v5; // edx
  unsigned int v7; // r8d
  unsigned int v8; // edi
  char result; // al
  int *v10; // rsi
  int v11; // ecx
  bool v12; // sf
  bool v13; // of
  int *v14; // rdx
  int *v15; // rax
  int v16; // edi
  int v17; // esi
  int v18; // ecx
  __int64 v19; // rdx
  int v20; // ecx
  int v21; // ebx
  unsigned int i; // eax
  _DWORD *v23; // r8
  unsigned int v24; // eax
  int v25; // eax
  int v26; // esi
  int v27; // edi
  int v28; // r10d
  unsigned int j; // eax
  _DWORD *v30; // r8
  unsigned int v31; // eax

  v4 = a2[63];
  v5 = a3[63];
  v7 = a3[64];
  v8 = a2[64];
  result = v8 == v7 && v4 == v5;
  if ( !result )
  {
    v10 = *a1;
    v11 = **a1;
    if ( v4 == v11 && v8 == v10[1] )
      return 1;
    if ( v5 == v11 && v7 == v10[1] )
    {
      if ( v4 == v5 )
        return v8 == v7;
      return result;
    }
    v13 = __OFSUB__(v4, v5);
    v12 = (int)(v4 - v5) < 0;
    if ( v4 == v5 )
      return v8 < v7;
    return v12 ^ v13;
  }
  v14 = a1[1];
  result = a2 == v14;
  if ( a2 == v14 || a3 == v14 )
    return result;
  if ( v4 )
  {
    v25 = a3[6];
    v13 = __OFSUB__(a2[6], v25);
    v12 = a2[6] - v25 < 0;
    return v12 ^ v13;
  }
  v15 = a1[2];
  v16 = a2[60];
  v17 = a2[61];
  v18 = v15[6];
  v19 = *((_QWORD *)v15 + 1);
  if ( !v18 )
    return 0;
  v20 = v18 - 1;
  v21 = 1;
  for ( i = v20
          & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v17) | ((unsigned __int64)(unsigned int)(37 * v16) << 32))) >> 31)
           ^ (756364221 * v17)); ; i = v20 & v24 )
  {
    v23 = (_DWORD *)(v19 + 24LL * i);
    if ( *v23 == v16 && v23[1] == v17 )
    {
      v4 = v23[5];
      v26 = a3[60];
      v27 = a3[61];
      goto LABEL_24;
    }
    if ( *v23 == -1 && v23[1] == -1 )
      break;
    v24 = v21 + i;
    ++v21;
  }
  v26 = a3[60];
  v27 = a3[61];
LABEL_24:
  v28 = 1;
  for ( j = v20
          & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v27) | ((unsigned __int64)(unsigned int)(37 * v26) << 32))) >> 31)
           ^ (756364221 * v27)); ; j = v20 & v31 )
  {
    v30 = (_DWORD *)(v19 + 24LL * j);
    if ( *v30 == v26 && v30[1] == v27 )
      return v30[5] > v4;
    if ( *v30 == -1 && v30[1] == -1 )
      break;
    v31 = v28 + j;
    ++v28;
  }
  return 0;
}

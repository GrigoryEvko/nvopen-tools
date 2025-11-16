// Function: sub_1377B70
// Address: 0x1377b70
//
__int64 __fastcall sub_1377B70(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned __int64 v8; // r10
  int v9; // r9d
  int v10; // r11d
  int v11; // r9d
  int v12; // r14d
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // r12
  unsigned int i; // esi
  __int64 v16; // r12
  unsigned int v17; // esi

  result = *(unsigned int *)(a1 + 48);
  if ( !(_DWORD)result )
    return result;
  v5 = *(_QWORD *)(a1 + 40);
  result = 3LL * *(unsigned int *)(a1 + 56);
  v6 = v5 + 24LL * *(unsigned int *)(a1 + 56);
  if ( v5 == v6 )
    return result;
  while ( 1 )
  {
    v7 = *(_QWORD *)v5;
    result = v5;
    if ( *(_QWORD *)v5 != -8 )
      break;
    if ( *(_DWORD *)(v5 + 8) != -1 )
      goto LABEL_6;
LABEL_35:
    v5 += 24;
    if ( v6 == v5 )
      return result;
  }
  if ( v7 == -16 && *(_DWORD *)(v5 + 8) == -2 )
    goto LABEL_35;
LABEL_6:
  if ( v6 == v5 )
    return result;
LABEL_11:
  if ( a2 != v7 || (v9 = *(_DWORD *)(a1 + 56)) == 0 )
  {
    result += 24;
    if ( result == v6 )
      return result;
    while ( 1 )
    {
      if ( *(_QWORD *)result == -8 )
      {
        if ( *(_DWORD *)(result + 8) != -1 )
          goto LABEL_9;
      }
      else if ( *(_QWORD *)result != -16 || *(_DWORD *)(result + 8) != -2 )
      {
LABEL_9:
        if ( v6 == result )
          return result;
        v7 = *(_QWORD *)result;
        goto LABEL_11;
      }
      result += 24;
      if ( v6 == result )
        return result;
    }
  }
LABEL_20:
  v10 = *(_DWORD *)(result + 8);
  v11 = v9 - 1;
  v12 = 1;
  v8 = (unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32;
  v13 = ((((unsigned int)(37 * v10) | v8) - 1 - ((unsigned __int64)(unsigned int)(37 * v10) << 32)) >> 22)
      ^ (((unsigned int)(37 * v10) | v8) - 1 - ((unsigned __int64)(unsigned int)(37 * v10) << 32));
  v14 = ((9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13)))) >> 15)
      ^ (9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13))));
  for ( i = v11 & (((v14 - 1 - (v14 << 27)) >> 31) ^ (v14 - 1 - ((_DWORD)v14 << 27))); ; i = v11 & v17 )
  {
    v16 = *(_QWORD *)(a1 + 40) + 24LL * i;
    if ( v7 == *(_QWORD *)v16 && *(_DWORD *)(v16 + 8) == v10 )
      break;
    if ( *(_QWORD *)v16 == -8 && *(_DWORD *)(v16 + 8) == -1 )
      goto LABEL_26;
    v17 = v12 + i;
    ++v12;
  }
  *(_QWORD *)v16 = -16;
  *(_DWORD *)(v16 + 8) = -2;
  --*(_DWORD *)(a1 + 48);
  ++*(_DWORD *)(a1 + 52);
LABEL_26:
  for ( result += 24; result != v6; result += 24 )
  {
    while ( 2 )
    {
      if ( *(_QWORD *)result == -8 )
      {
        if ( *(_DWORD *)(result + 8) == -1 )
          goto LABEL_38;
      }
      else if ( *(_QWORD *)result == -16 && *(_DWORD *)(result + 8) == -2 )
      {
LABEL_38:
        result += 24;
        if ( v6 == result )
          return result;
        continue;
      }
      break;
    }
    if ( v6 == result )
      return result;
    v7 = *(_QWORD *)result;
    if ( a2 != *(_QWORD *)result )
      goto LABEL_26;
    v9 = *(_DWORD *)(a1 + 56);
    if ( v9 )
      goto LABEL_20;
  }
  return result;
}

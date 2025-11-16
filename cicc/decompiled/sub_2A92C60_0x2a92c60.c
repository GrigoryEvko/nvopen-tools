// Function: sub_2A92C60
// Address: 0x2a92c60
//
__int64 __fastcall sub_2A92C60(__int64 a1, char *a2, _QWORD *a3)
{
  int v3; // r10d
  int v5; // r10d
  __int64 v6; // r12
  int v7; // eax
  __int64 v8; // r9
  int v9; // edi
  char v10; // r8
  int v11; // esi
  unsigned __int64 v12; // rdx
  int v13; // ebx
  unsigned int i; // eax
  __int64 v15; // rdx
  int v16; // r11d
  unsigned int v17; // eax

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v3 - 1;
  v6 = 0;
  v7 = *a2;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *((_DWORD *)a2 + 1);
  v10 = *a2;
  v11 = *((_DWORD *)a2 + 2);
  v12 = (unsigned int)(1512728442 * v7) | ((unsigned __int64)(unsigned int)(37 * v9) << 32);
  v13 = 1;
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)((0xBF58476D1CE4E5B9LL * v12) >> 31) ^ (484763065 * (_DWORD)v12)
              | ((unsigned __int64)(unsigned int)(37 * v11) << 32))) >> 31)
           ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v12) >> 31) ^ (484763065 * v12)))); ; i = v5 & v17 )
  {
    v15 = v8 + 16LL * i;
    v16 = *(_DWORD *)(v15 + 8);
    if ( v16 == v11 && v9 == *(_DWORD *)(v15 + 4) && v10 == *(_BYTE *)v15 )
    {
      *a3 = v15;
      return 1;
    }
    if ( v16 == -1 )
      break;
    if ( v16 == -2 && *(_DWORD *)(v15 + 4) == -2 && *(_BYTE *)v15 == 0xFE && !v6 )
      v6 = v8 + 16LL * i;
LABEL_7:
    v17 = v13 + i;
    ++v13;
  }
  if ( *(_DWORD *)(v15 + 4) != -1 || *(_BYTE *)v15 != 0xFF )
    goto LABEL_7;
  if ( !v6 )
    v6 = v8 + 16LL * i;
  *a3 = v6;
  return 0;
}

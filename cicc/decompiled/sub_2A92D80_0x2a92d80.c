// Function: sub_2A92D80
// Address: 0x2a92d80
//
__int64 __fastcall sub_2A92D80(__int64 a1, char *a2, _QWORD *a3)
{
  int v3; // r11d
  int v6; // r11d
  int v7; // eax
  __int64 v8; // r9
  int v9; // edi
  char v10; // r8
  int v11; // esi
  __int64 v12; // r10
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rdx
  __int64 v15; // r12
  int v16; // ebx
  unsigned int i; // eax
  __int64 v18; // rdx
  __int64 v19; // r13
  unsigned int v20; // eax

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = v3 - 1;
  v7 = *a2;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *((_DWORD *)a2 + 1);
  v10 = *a2;
  v11 = *((_DWORD *)a2 + 2);
  v12 = *((_QWORD *)a2 + 2);
  v13 = (unsigned int)(1512728442 * v7) | ((unsigned __int64)(unsigned int)(37 * v9) << 32);
  v14 = 0xBF58476D1CE4E5B9LL
      * (((unsigned __int64)(unsigned int)(37 * v11) << 32)
       | (unsigned int)((0xBF58476D1CE4E5B9LL * v13) >> 31) ^ (484763065 * (_DWORD)v13));
  v15 = 0;
  v16 = 1;
  for ( i = v6
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)(v14 >> 31) ^ (unsigned int)v14
              | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))) >> 31)
           ^ (484763065 * ((v14 >> 31) ^ v14))); ; i = v6 & v20 )
  {
    v18 = v8 + 32LL * i;
    v19 = *(_QWORD *)(v18 + 16);
    if ( v12 == v19 && v11 == *(_DWORD *)(v18 + 8) && v9 == *(_DWORD *)(v18 + 4) && v10 == *(_BYTE *)v18 )
    {
      *a3 = v18;
      return 1;
    }
    if ( v19 == -4096 )
      break;
    if ( v19 == -8192 && *(_DWORD *)(v18 + 8) == -2 && *(_DWORD *)(v18 + 4) == -2 && *(_BYTE *)v18 == 0xFE && !v15 )
      v15 = v8 + 32LL * i;
LABEL_7:
    v20 = v16 + i;
    ++v16;
  }
  if ( *(_DWORD *)(v18 + 8) != -1 || *(_DWORD *)(v18 + 4) != -1 || *(_BYTE *)v18 != 0xFF )
    goto LABEL_7;
  if ( !v15 )
    v15 = v8 + 32LL * i;
  *a3 = v15;
  return 0;
}

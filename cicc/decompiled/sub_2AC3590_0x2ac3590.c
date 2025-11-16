// Function: sub_2AC3590
// Address: 0x2ac3590
//
__int64 __fastcall sub_2AC3590(__int64 a1, __int64 *a2)
{
  int v2; // r9d
  __int64 v4; // rsi
  int v5; // edx
  char v6; // di
  int v7; // r11d
  __int64 v8; // rcx
  int v9; // r9d
  unsigned int i; // eax
  __int64 v11; // r8
  unsigned int v12; // eax

  v2 = *(_DWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  if ( !v2 )
    return 0;
  v5 = *((_DWORD *)a2 + 2);
  v6 = *((_BYTE *)a2 + 12);
  v7 = 1;
  v8 = *a2;
  v9 = v2 - 1;
  for ( i = v9
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)(v6 == 0) + 37 * v5 - 1)
              | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))) >> 31)
           ^ (484763065 * ((v6 == 0) + 37 * v5 - 1))); ; i = v9 & v12 )
  {
    v11 = v4 + ((unsigned __int64)i << 6);
    if ( *(_QWORD *)v11 == v8 && v5 == *(_DWORD *)(v11 + 8) && v6 == *(_BYTE *)(v11 + 12) )
      break;
    if ( *(_QWORD *)v11 == -4096 && *(_DWORD *)(v11 + 8) == -1 && *(_BYTE *)(v11 + 12) )
      return 0;
    v12 = v7 + i;
    ++v7;
  }
  return v4 + ((unsigned __int64)i << 6);
}

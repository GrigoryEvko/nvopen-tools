// Function: sub_2A75400
// Address: 0x2a75400
//
__int64 __fastcall sub_2A75400(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v3; // ecx
  int v5; // ecx
  __int64 v6; // r11
  __int64 v7; // r9
  int v8; // ebx
  __int64 v9; // r8
  __int64 v10; // rdi
  unsigned int i; // eax
  __int64 v12; // rsi
  __int64 v13; // r10
  unsigned int v14; // eax

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v3 - 1;
  v6 = 0;
  v7 = *(_QWORD *)(a2 + 16);
  v8 = 1;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_QWORD *)(a2 + 40);
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)
              | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)))); ; i = v5 & v14 )
  {
    v12 = v9 + 80LL * i;
    v13 = *(_QWORD *)(v12 + 16);
    if ( v13 == v7 && *(_QWORD *)(v12 + 40) == v10 )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -4096 )
      break;
    if ( v13 == -8192 && *(_QWORD *)(v12 + 40) == -8192 && !v6 )
      v6 = v9 + 80LL * i;
LABEL_10:
    v14 = v8 + i;
    ++v8;
  }
  if ( *(_QWORD *)(v12 + 40) != -4096 )
    goto LABEL_10;
  if ( !v6 )
    v6 = v9 + 80LL * i;
  *a3 = v6;
  return 0;
}

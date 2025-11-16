// Function: sub_2A68270
// Address: 0x2a68270
//
__int64 __fastcall sub_2A68270(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // edx
  int v5; // edx
  __int64 v6; // r11
  int v7; // ecx
  int v8; // ebx
  __int64 v9; // rsi
  __int64 v10; // rdi
  unsigned int i; // eax
  __int64 *v12; // r9
  __int64 v13; // r10
  unsigned int v14; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v4 - 1;
  v6 = 0;
  v7 = *((_DWORD *)a2 + 2);
  v8 = 1;
  v9 = *a2;
  v10 = *(_QWORD *)(a1 + 8);
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)(37 * v7) | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))) >> 31)
           ^ (756364221 * v7)); ; i = v5 & v14 )
  {
    v12 = (__int64 *)(v10 + 24LL * i);
    v13 = *v12;
    if ( *v12 == v9 && *((_DWORD *)v12 + 2) == v7 )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -4096 )
      break;
    if ( v13 == -8192 && *((_DWORD *)v12 + 2) == -2 && !v6 )
      v6 = v10 + 24LL * i;
LABEL_9:
    v14 = v8 + i;
    ++v8;
  }
  if ( *((_DWORD *)v12 + 2) != -1 )
    goto LABEL_9;
  if ( !v6 )
    v6 = v10 + 24LL * i;
  *a3 = v6;
  return 0;
}

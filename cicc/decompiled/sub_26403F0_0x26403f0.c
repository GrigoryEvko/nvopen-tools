// Function: sub_26403F0
// Address: 0x26403f0
//
__int64 __fastcall sub_26403F0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // edx
  int v5; // r10d
  __int64 v6; // r11
  __int64 v7; // r8
  int v8; // ebx
  int v9; // edi
  __int64 v10; // rsi
  unsigned int i; // eax
  __int64 *v12; // rdx
  __int64 v13; // r9
  unsigned int v14; // eax

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v4 - 1;
  v6 = 0;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = *((_DWORD *)a2 + 2);
  v10 = *a2;
  for ( i = (v4 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)(37 * v9) | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))) >> 31)
           ^ (756364221 * v9)); ; i = v5 & v14 )
  {
    v12 = (__int64 *)(v7 + 32LL * i);
    v13 = *v12;
    if ( *v12 == v10 && *((_DWORD *)v12 + 2) == v9 )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -4096 )
      break;
    if ( v13 == -8192 && *((_DWORD *)v12 + 2) == -2 && !v6 )
      v6 = v7 + 32LL * i;
LABEL_9:
    v14 = v8 + i;
    ++v8;
  }
  if ( *((_DWORD *)v12 + 2) != -1 )
    goto LABEL_9;
  if ( !v6 )
    v6 = v7 + 32LL * i;
  *a3 = v6;
  return 0;
}

// Function: sub_2A65F30
// Address: 0x2a65f30
//
__int64 __fastcall sub_2A65F30(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v3; // r9d
  int v4; // r9d
  __int64 v5; // r11
  __int64 v6; // r8
  int v7; // ebx
  int v8; // ecx
  __int64 v9; // rdi
  unsigned int i; // eax
  __int64 *v11; // rsi
  __int64 v12; // r10
  unsigned int v13; // eax

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v4 = v3 - 1;
  v5 = 0;
  v6 = *a2;
  v7 = 1;
  v8 = *((_DWORD *)a2 + 2);
  v9 = *(_QWORD *)(a1 + 8);
  for ( i = v4
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)(37 * v8) | ((unsigned __int64)(((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4)) << 32))) >> 31)
           ^ (756364221 * v8)); ; i = v4 & v13 )
  {
    v11 = (__int64 *)(v9 + 56LL * i);
    v12 = *v11;
    if ( *v11 == v6 && *((_DWORD *)v11 + 2) == v8 )
    {
      *a3 = v11;
      return 1;
    }
    if ( v12 == -4096 )
      break;
    if ( v12 == -8192 && *((_DWORD *)v11 + 2) == -2 && !v5 )
      v5 = v9 + 56LL * i;
LABEL_9:
    v13 = v7 + i;
    ++v7;
  }
  if ( *((_DWORD *)v11 + 2) != -1 )
    goto LABEL_9;
  if ( !v5 )
    v5 = v9 + 56LL * i;
  *a3 = v5;
  return 0;
}

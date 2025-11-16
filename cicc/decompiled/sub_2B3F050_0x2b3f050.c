// Function: sub_2B3F050
// Address: 0x2b3f050
//
__int64 __fastcall sub_2B3F050(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v3; // r10d
  int v6; // r10d
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdi
  int v10; // ebx
  __int64 *v11; // r11
  unsigned int i; // eax
  __int64 *v13; // rdx
  __int64 v14; // rcx
  unsigned int v15; // eax

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = v3 - 1;
  v7 = *a2;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = a2[1];
  v10 = 1;
  v11 = 0;
  for ( i = v6
          & (((0xBF58476D1CE4E5B9LL
             * (((((0xBF58476D1CE4E5B9LL * *a2) >> 31) ^ (0xBF58476D1CE4E5B9LL * *a2)) << 32)
              | ((unsigned int)v9 >> 4) ^ ((unsigned int)v9 >> 9))) >> 31)
           ^ (484763065 * (((unsigned int)v9 >> 4) ^ ((unsigned int)v9 >> 9)))); ; i = v6 & v15 )
  {
    v13 = (__int64 *)(v8 + 80LL * i);
    v14 = *v13;
    if ( *v13 == v7 && v13[1] == v9 )
    {
      *a3 = v13;
      return 1;
    }
    if ( v14 == -1 )
      break;
    if ( v14 == -2 && v13[1] == -8192 && !v11 )
      v11 = (__int64 *)(v8 + 80LL * i);
LABEL_9:
    v15 = v10 + i;
    ++v10;
  }
  if ( v13[1] != -4096 )
    goto LABEL_9;
  if ( !v11 )
    v11 = (__int64 *)(v8 + 80LL * i);
  *a3 = v11;
  return 0;
}

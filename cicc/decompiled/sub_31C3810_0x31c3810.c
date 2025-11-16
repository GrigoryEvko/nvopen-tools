// Function: sub_31C3810
// Address: 0x31c3810
//
__int64 __fastcall sub_31C3810(__int64 a1, int *a2, _QWORD *a3)
{
  int v3; // r10d
  int v7; // r10d
  int v8; // esi
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11; // rdi
  __int64 v12; // r12
  int v13; // ebx
  unsigned int i; // eax
  __int64 v15; // rdx
  __int64 v16; // r11
  unsigned int v17; // eax

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v7 = v3 - 1;
  v8 = *a2;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *((_QWORD *)a2 + 1);
  v11 = *((_QWORD *)a2 + 2);
  v12 = 0;
  v13 = 1;
  for ( i = v7
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)((0xBF58476D1CE4E5B9LL
                              * ((unsigned int)(1512728442 * v8)
                               | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))) >> 31)
              ^ (-1747130070 * v8)
              | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))) >> 31)
           ^ (484763065
            * (((0xBF58476D1CE4E5B9LL
               * ((unsigned int)(1512728442 * v8)
                | ((unsigned __int64)(((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)) << 32))) >> 31)
             ^ (-1747130070 * v8)))); ; i = v7 & v17 )
  {
    v15 = v9 + 32LL * i;
    v16 = *(_QWORD *)(v15 + 16);
    if ( v16 == v11 && v10 == *(_QWORD *)(v15 + 8) && v8 == *(_DWORD *)v15 )
    {
      *a3 = v15;
      return 1;
    }
    if ( v16 == -4096 )
      break;
    if ( v16 == -8192 && *(_QWORD *)(v15 + 8) == -8192 && *(_DWORD *)v15 == 0x80000000 && !v12 )
      v12 = v9 + 32LL * i;
LABEL_7:
    v17 = v13 + i;
    ++v13;
  }
  if ( *(_QWORD *)(v15 + 8) != -4096 || *(_DWORD *)v15 != 0x7FFFFFFF )
    goto LABEL_7;
  if ( !v12 )
    v12 = v9 + 32LL * i;
  *a3 = v12;
  return 0;
}

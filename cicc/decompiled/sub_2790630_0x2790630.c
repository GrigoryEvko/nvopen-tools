// Function: sub_2790630
// Address: 0x2790630
//
__int64 __fastcall sub_2790630(__int64 a1, int *a2, _QWORD *a3)
{
  int v4; // edx
  int v5; // edx
  __int64 v6; // r11
  __int64 v7; // r8
  int v8; // ebx
  __int64 v9; // rdi
  int v10; // esi
  unsigned int i; // eax
  int *v12; // r9
  int v13; // r10d
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
  v9 = *((_QWORD *)a2 + 1);
  v10 = *a2;
  for ( i = v5
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned __int64)(unsigned int)(37 * v10) << 32) | ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4))) >> 31)
           ^ (484763065 * (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)))); ; i = v5 & v14 )
  {
    v12 = (int *)(v7 + 24LL * i);
    v13 = *v12;
    if ( *v12 == v10 && *((_QWORD *)v12 + 1) == v9 )
    {
      *a3 = v12;
      return 1;
    }
    if ( v13 == -1 )
      break;
    if ( v13 == -2 && *((_QWORD *)v12 + 1) == -8192 && !v6 )
      v6 = v7 + 24LL * i;
LABEL_9:
    v14 = v8 + i;
    ++v8;
  }
  if ( *((_QWORD *)v12 + 1) != -4096 )
    goto LABEL_9;
  if ( !v6 )
    v6 = v7 + 24LL * i;
  *a3 = v6;
  return 0;
}

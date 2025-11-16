// Function: sub_2AAA2B0
// Address: 0x2aaa2b0
//
__int64 __fastcall sub_2AAA2B0(__int64 a1, __int64 a2, int a3, char a4)
{
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // ebx
  unsigned int i; // eax
  __int64 v12; // r9
  unsigned int v13; // eax

  v8 = *(unsigned int *)(a1 + 376);
  v9 = *(_QWORD *)(a1 + 360);
  if ( !(_DWORD)v8 )
    return 0;
  v10 = 1;
  for ( i = (v8 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)(a4 == 0) + 37 * a3 - 1)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * ((a4 == 0) + 37 * a3 - 1))); ; i = (v8 - 1) & v13 )
  {
    v12 = v9 + 40LL * i;
    if ( a2 == *(_QWORD *)v12 && a3 == *(_DWORD *)(v12 + 8) && a4 == *(_BYTE *)(v12 + 12) )
      break;
    if ( *(_QWORD *)v12 == -4096 && *(_DWORD *)(v12 + 8) == -1 && *(_BYTE *)(v12 + 12) )
      return 0;
    v13 = v10 + i;
    ++v10;
  }
  if ( v12 != v9 + 40 * v8 )
    return *(unsigned int *)(v12 + 16);
  return 0;
}

// Function: sub_227BA70
// Address: 0x227ba70
//
__int64 __fastcall sub_227BA70(__int64 a1, __int64 *a2)
{
  int v2; // r8d
  __int64 v3; // rcx
  __int64 v4; // rdi
  __int64 v5; // rsi
  int v6; // r8d
  int v7; // r10d
  unsigned int i; // eax
  _QWORD *v9; // r9
  unsigned int v10; // eax

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( !v2 )
    return 0;
  v4 = *a2;
  v5 = a2[1];
  v6 = v2 - 1;
  v7 = 1;
  for ( i = v6
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
              | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; i = v6 & v10 )
  {
    v9 = (_QWORD *)(v3 + 24LL * i);
    if ( *v9 == v4 && v9[1] == v5 )
      break;
    if ( *v9 == -4096 && v9[1] == -4096 )
      return 0;
    v10 = v7 + i;
    ++v7;
  }
  return v3 + 24LL * i;
}

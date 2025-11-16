// Function: sub_297C3E0
// Address: 0x297c3e0
//
_QWORD *__fastcall sub_297C3E0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  int v9; // ebx
  unsigned int i; // eax
  _QWORD *v11; // rdx
  __int64 v12; // r10
  unsigned int v13; // eax

  v3 = a2[4];
  v4 = a2[7];
  v5 = *(unsigned int *)(a1 + 24);
  v6 = a2[1];
  v7 = *(_QWORD *)(v3 + 8);
  v8 = *(_QWORD *)(a1 + 8);
  if ( !(_DWORD)v5 )
    return 0;
  v9 = 1;
  for ( i = (v5 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)((0xBF58476D1CE4E5B9LL
                              * ((969526130 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))
                               | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))) >> 31)
              ^ (-279380126 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))
              | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
           ^ (484763065
            * (((0xBF58476D1CE4E5B9LL
               * ((969526130 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))
                | ((unsigned __int64)(((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4)) << 32))) >> 31)
             ^ (-279380126 * (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)))))); ; i = (v5 - 1) & v13 )
  {
    v11 = (_QWORD *)(v8 + 56LL * i);
    v12 = v11[2];
    if ( v6 == v12 && v4 == v11[1] && v7 == *v11 )
      break;
    if ( v12 == -4096 && v11[1] == -4096 && *v11 == -4096 )
      return 0;
    v13 = v9 + i;
    ++v9;
  }
  if ( v11 != (_QWORD *)(v8 + 56 * v5) )
    return v11 + 3;
  return 0;
}

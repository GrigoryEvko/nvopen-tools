// Function: sub_2A63A20
// Address: 0x2a63a20
//
__int64 __fastcall sub_2A63A20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  int v6; // r8d
  int v7; // r8d
  int v8; // r10d
  unsigned int i; // eax
  _QWORD *v10; // rdx
  unsigned int v11; // eax

  v5 = *(_QWORD *)(a1 + 2512);
  v6 = *(_DWORD *)(a1 + 2528);
  if ( !v6 )
    return 0;
  v7 = v6 - 1;
  v8 = 1;
  for ( i = v7
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = v7 & v11 )
  {
    v10 = (_QWORD *)(v5 + 16LL * i);
    if ( a2 == *v10 && a3 == v10[1] )
      break;
    if ( *v10 == -4096 && v10[1] == -4096 )
      return 0;
    v11 = v8 + i;
    ++v8;
  }
  return 1;
}

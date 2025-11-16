// Function: sub_2950690
// Address: 0x2950690
//
__int64 __fastcall sub_2950690(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rcx
  __int64 v8; // rdi
  int v9; // r10d
  unsigned int i; // eax
  __int64 v11; // rbx
  unsigned int v12; // eax
  __int64 v13; // rax
  __int64 v14; // r12

  v7 = *(unsigned int *)(a5 + 24);
  v8 = *(_QWORD *)(a5 + 8);
  if ( !(_DWORD)v7 )
    return 0;
  v9 = 1;
  for ( i = (v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v7 - 1) & v12 )
  {
    v11 = v8 + 48LL * i;
    if ( *(_QWORD *)v11 == a2 && *(_QWORD *)(v11 + 8) == a3 )
      break;
    if ( *(_QWORD *)v11 == -4096 && *(_QWORD *)(v11 + 8) == -4096 )
      return 0;
    v12 = v9 + i;
    ++v9;
  }
  if ( v11 == 48 * v7 + v8 )
    return 0;
  v13 = *(unsigned int *)(v11 + 24);
  if ( !(_DWORD)v13 )
    return 0;
  while ( 1 )
  {
    v14 = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 8 * v13 - 8);
    if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 8), v14, a4) )
      break;
    v13 = (unsigned int)(*(_DWORD *)(v11 + 24) - 1);
    *(_DWORD *)(v11 + 24) = v13;
    if ( !(_DWORD)v13 )
      return 0;
  }
  return v14;
}

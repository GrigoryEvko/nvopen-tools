// Function: sub_33512A0
// Address: 0x33512a0
//
__int64 __fastcall sub_33512A0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  int v6; // eax
  unsigned int *v7; // rax
  unsigned int *v8; // rcx
  __int64 result; // rax
  int v10; // eax
  _QWORD *v11; // r15
  _QWORD *v12; // r14

  if ( a1 == a2 )
    return 1;
  while ( 1 )
  {
    v6 = *(_DWORD *)(a1 + 24);
    if ( v6 == 2 )
      break;
    if ( v6 < 0 )
    {
      v10 = ~v6;
      if ( *(_DWORD *)(a4 + 68) == v10 )
      {
        ++a3;
      }
      else if ( v10 == *(_DWORD *)(a4 + 64) )
      {
        if ( !a3 )
          return 0;
        --a3;
      }
    }
    v7 = *(unsigned int **)(a1 + 40);
    v8 = &v7[10 * *(unsigned int *)(a1 + 64)];
    if ( v7 == v8 )
      return 0;
    while ( 1 )
    {
      a1 = *(_QWORD *)v7;
      if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v7 + 48LL) + 16LL * v7[2]) == 1 )
        break;
      v7 += 10;
      if ( v8 == v7 )
        return 0;
    }
    if ( *(_DWORD *)(a1 + 24) == 1 )
      return 0;
    if ( a1 == a2 )
      return 1;
  }
  v11 = *(_QWORD **)(a1 + 40);
  v12 = &v11[5 * *(unsigned int *)(a1 + 64)];
  if ( v12 == v11 )
    return 0;
  do
  {
    result = sub_33512A0(*v11, a2, a3, a4);
    if ( (_BYTE)result )
      break;
    v11 += 5;
  }
  while ( v12 != v11 );
  return result;
}

// Function: sub_C50DF0
// Address: 0xc50df0
//
__int64 __fastcall sub_C50DF0(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 *a3,
        unsigned __int8 (__fastcall *a4)(_QWORD),
        __int64 a5)
{
  unsigned __int64 v5; // r15
  __int64 v7; // r12
  unsigned int v9; // eax
  int v10; // eax
  __int64 v11; // r14

  v5 = a2;
  v7 = a1;
  while ( 1 )
  {
    v9 = sub_C92610(a1, a2);
    v10 = sub_C92860(a5, v7, v5, v9);
    if ( v10 != -1 )
    {
      v11 = *(_QWORD *)a5 + 8LL * v10;
      if ( v11 != *(_QWORD *)a5 + 8LL * *(unsigned int *)(a5 + 8)
        && a4(*(_QWORD *)(*(_QWORD *)v11 + 8LL))
        && v11 != *(_QWORD *)a5 + 8LL * *(unsigned int *)(a5 + 8) )
      {
        break;
      }
    }
    if ( v5 <= 1 )
      return 0;
    --v5;
    a1 = v7;
    a2 = v5;
  }
  if ( a4(*(_QWORD *)(*(_QWORD *)v11 + 8LL)) )
  {
    *a3 = v5;
    return *(_QWORD *)(*(_QWORD *)v11 + 8LL);
  }
  return 0;
}

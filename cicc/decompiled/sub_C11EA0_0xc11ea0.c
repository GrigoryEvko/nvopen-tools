// Function: sub_C11EA0
// Address: 0xc11ea0
//
_QWORD *__fastcall sub_C11EA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  _QWORD *result; // rax
  _QWORD *v4; // r12
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rdx

  sub_C152E0(a2);
  v2 = sub_C14A60(a2);
  result = (_QWORD *)sub_C14AB0(a2);
  if ( (_QWORD *)v2 != result )
  {
    v4 = result;
    do
    {
      while ( 1 )
      {
        switch ( *(_DWORD *)(*(_QWORD *)v2 + 8LL) )
        {
          case 0:
            BUG();
          case 1:
          case 5:
            v5 = 2051;
            break;
          case 3:
            v5 = 2050;
            break;
          case 4:
            v5 = 2054;
            break;
          case 6:
            v5 = 2053;
            break;
          default:
            v5 = 2048;
            break;
        }
        (**(void (__fastcall ***)(_QWORD, __int64, _QWORD, __int64))a1)(
          *(_QWORD *)(*(_QWORD *)a1 + 8LL),
          *(_QWORD *)v2 + 16LL,
          **(_QWORD **)v2,
          v5);
        result = *(_QWORD **)(v2 + 8);
        v6 = v2 + 8;
        if ( result == (_QWORD *)-8LL || !result )
          break;
        v2 += 8;
        if ( (_QWORD *)v6 == v4 )
          return result;
      }
      result = (_QWORD *)(v2 + 16);
      do
      {
        do
        {
          v7 = *result;
          v2 = (__int64)result++;
        }
        while ( !v7 );
      }
      while ( v7 == -8 );
    }
    while ( (_QWORD *)v2 != v4 );
  }
  return result;
}

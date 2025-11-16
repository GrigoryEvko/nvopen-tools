// Function: sub_D23BB0
// Address: 0xd23bb0
//
_QWORD *__fastcall sub_D23BB0(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *v2; // rsi

  result = *(_QWORD **)a1;
  v2 = *(_QWORD **)(a1 + 8);
  if ( v2 != *(_QWORD **)a1 )
  {
    do
    {
      if ( (*result & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(_QWORD *)(*result & 0xFFFFFFFFFFFFFFF8LL) && (*result & 4) != 0 )
        break;
      *(_QWORD *)a1 = ++result;
    }
    while ( result != v2 );
  }
  return result;
}

// Function: sub_35BB790
// Address: 0x35bb790
//
_QWORD *__fastcall sub_35BB790(__int64 a1, _QWORD *a2)
{
  _QWORD *result; // rax

  result = *(_QWORD **)(a1 + 8);
  if ( result == *(_QWORD **)(a1 + 16) )
  {
    sub_35BB5E0((unsigned __int64 *)a1, *(char **)(a1 + 8), a2);
    return (_QWORD *)(*(_QWORD *)(a1 + 8) - 8LL);
  }
  else
  {
    if ( result )
    {
      *result = *a2;
      *a2 = 0;
      result = *(_QWORD **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = result + 1;
  }
  return result;
}

// Function: sub_19E4730
// Address: 0x19e4730
//
_QWORD *__fastcall sub_19E4730(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *v2; // rdx

  result = *(_QWORD **)a1;
  v2 = *(_QWORD **)(a1 + 8);
  if ( *(_QWORD **)a1 != v2 )
  {
    do
    {
      if ( *result < 0xFFFFFFFFFFFFFFFELL )
        break;
      *(_QWORD *)a1 = ++result;
    }
    while ( result != v2 );
  }
  return result;
}

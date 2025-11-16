// Function: sub_732D20
// Address: 0x732d20
//
_QWORD *__fastcall sub_732D20(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  _QWORD *result; // rax
  char v5; // al

  if ( a4 )
  {
    result = (_QWORD *)*a4;
  }
  else
  {
    if ( a3 )
    {
      a2 = *(_QWORD *)(*(_QWORD *)(a3 + 168) + 152LL);
    }
    else if ( !a2 || ((v5 = *(_BYTE *)(a2 + 28), v5 == 2) || v5 == 17) && (*(_BYTE *)(a1 - 8) & 1) != 0 )
    {
      a2 = unk_4F07288;
    }
    result = *(_QWORD **)(a2 + 232);
  }
  for ( ; result; result = (_QWORD *)*result )
  {
    if ( result[3] == a1 )
      break;
  }
  return result;
}

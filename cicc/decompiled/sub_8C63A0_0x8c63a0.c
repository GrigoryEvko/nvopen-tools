// Function: sub_8C63A0
// Address: 0x8c63a0
//
_QWORD *__fastcall sub_8C63A0(_QWORD *a1)
{
  _QWORD *result; // rax
  __int64 v2; // rcx
  __int64 v3; // rsi

  for ( result = a1; result; result = (_QWORD *)*result )
  {
    v2 = result[1];
    if ( (unsigned __int8)(*(_BYTE *)(v2 + 140) - 9) > 2u )
      break;
    v3 = *(_QWORD *)(v2 + 168);
    if ( (!v3 || (*(_BYTE *)(v3 + 109) & 8) == 0) && ((*(_BYTE *)(v2 + 177) & 0x90) != 0x10 || !*(_QWORD *)(v3 + 168)) )
      break;
  }
  return result;
}

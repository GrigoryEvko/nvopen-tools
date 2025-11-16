// Function: sub_8C62C0
// Address: 0x8c62c0
//
_QWORD *__fastcall sub_8C62C0(_QWORD *a1)
{
  _QWORD *result; // rax
  __int64 v2; // rdx

  for ( result = a1; result; result = (_QWORD *)*result )
  {
    v2 = result[1];
    if ( ((*(_BYTE *)(v2 + 193) & 0x10) == 0 || !*(_QWORD *)(v2 + 280))
      && ((*(_BYTE *)(v2 + 195) & 1) == 0 || !*(_QWORD *)(v2 + 240)) )
    {
      break;
    }
  }
  return result;
}

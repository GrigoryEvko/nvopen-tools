// Function: sub_8C3FF0
// Address: 0x8c3ff0
//
_QWORD *__fastcall sub_8C3FF0(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v2; // rdx

  result = *(_QWORD **)(a1 + 32);
  if ( result && *result == a1 )
  {
    v2 = result[1];
    if ( v2 )
      *result = v2;
  }
  return result;
}

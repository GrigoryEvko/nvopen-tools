// Function: sub_824D70
// Address: 0x824d70
//
__int64 *__fastcall sub_824D70(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  _QWORD *i; // rdx

  result = *(__int64 **)(a2 + 64);
  for ( i = (_QWORD *)(a2 + 64); result; result = (__int64 *)*result )
  {
    if ( result[2] == a1 && (result[6] & 1) == 0 )
    {
      *i = *result;
      sub_721090();
    }
    i = result;
  }
  return result;
}

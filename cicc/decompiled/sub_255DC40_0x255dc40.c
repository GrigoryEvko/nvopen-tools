// Function: sub_255DC40
// Address: 0x255dc40
//
__int64 *__fastcall sub_255DC40(__int64 a1)
{
  __int64 *result; // rax
  __int64 *v2; // rsi
  __int64 v3; // rdx

  result = *(__int64 **)(a1 + 16);
  v2 = *(__int64 **)(a1 + 24);
  if ( result != v2 )
  {
    v3 = *result;
    if ( *result == 0x7FFFFFFFFFFFFFFFLL )
      goto LABEL_8;
LABEL_3:
    if ( v3 == 0x7FFFFFFFFFFFFFFELL && result[1] == 0x7FFFFFFFFFFFFFFELL )
    {
      do
      {
        result += 12;
        *(_QWORD *)(a1 + 16) = result;
        if ( result == v2 )
          break;
        v3 = *result;
        if ( *result != 0x7FFFFFFFFFFFFFFFLL )
          goto LABEL_3;
LABEL_8:
        ;
      }
      while ( result[1] == 0x7FFFFFFFFFFFFFFFLL );
    }
  }
  return result;
}

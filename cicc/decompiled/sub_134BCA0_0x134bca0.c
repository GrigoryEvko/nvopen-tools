// Function: sub_134BCA0
// Address: 0x134bca0
//
char __fastcall sub_134BCA0(_QWORD *a1, _BYTE *a2)
{
  char result; // al

  a2[35] = 1;
  result = sub_134BA10(a1, (__int64)a2);
  if ( !a2[18] )
  {
    if ( !a2[19] )
      return result;
    return sub_134B500((__int64)a1, (__int64)a2);
  }
  result = sub_134B650((__int64)a1, (__int64)a2);
  if ( a2[19] )
    return sub_134B500((__int64)a1, (__int64)a2);
  return result;
}

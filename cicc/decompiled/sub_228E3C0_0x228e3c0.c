// Function: sub_228E3C0
// Address: 0x228e3c0
//
_QWORD *__fastcall sub_228E3C0(__int64 a1, char *a2, __int64 a3)
{
  _QWORD *result; // rax

  result = sub_228E360(a1, a2, a3);
  if ( result )
  {
    if ( *((_WORD *)result + 12) )
      return 0;
  }
  return result;
}

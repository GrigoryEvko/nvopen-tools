// Function: sub_31770C0
// Address: 0x31770c0
//
unsigned __int8 *__fastcall sub_31770C0(__int64 **a1, _BYTE *a2, unsigned __int8 *a3)
{
  unsigned __int8 *result; // rax

  if ( !a3 )
    return 0;
  result = sub_BD3990(a3, (__int64)a2);
  if ( *result != 17 )
  {
    if ( *result == 60 && *(_BYTE *)(*((_QWORD *)result + 9) + 8LL) == 12 )
      return (unsigned __int8 *)sub_3177070(a1, (__int64)result, a2);
    else
      return 0;
  }
  return result;
}

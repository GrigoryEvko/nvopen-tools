// Function: sub_27DB860
// Address: 0x27db860
//
unsigned __int8 *__fastcall sub_27DB860(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *result; // rax
  int v3; // ecx

  if ( !a1 )
    return 0;
  result = a1;
  v3 = *a1;
  if ( (unsigned int)(v3 - 12) <= 1 )
    return result;
  if ( (_DWORD)a2 != 1 )
  {
    if ( (_BYTE)v3 == 17 )
      return result;
    return 0;
  }
  result = sub_BD3990(a1, a2);
  if ( *result != 4 )
    return 0;
  return result;
}

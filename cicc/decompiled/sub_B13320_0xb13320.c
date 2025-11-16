// Function: sub_B13320
// Address: 0xb13320
//
unsigned __int8 *__fastcall sub_B13320(__int64 a1)
{
  unsigned __int8 *result; // rax

  if ( *(_BYTE *)(a1 + 64) == 2 )
    result = *(unsigned __int8 **)(a1 + 48);
  else
    result = *(unsigned __int8 **)(a1 + 40);
  if ( result )
  {
    if ( (unsigned int)*result - 1 > 1 )
      return 0;
    else
      return (unsigned __int8 *)*((_QWORD *)result + 17);
  }
  return result;
}

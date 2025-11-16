// Function: sub_3506EB0
// Address: 0x3506eb0
//
unsigned __int64 *__fastcall sub_3506EB0(unsigned __int64 **a1, __int64 a2)
{
  unsigned __int64 *result; // rax
  unsigned __int64 v3; // rcx

  result = *a1;
  v3 = **a1;
  if ( (*(_BYTE *)(v3 + 3) & 0x10) != 0 )
    return (unsigned __int64 *)sub_3506D90(result[1], (__int64 *)result[2], a2, v3);
  return result;
}

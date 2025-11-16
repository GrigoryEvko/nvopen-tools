// Function: sub_22410F0
// Address: 0x22410f0
//
unsigned __int64 *__fastcall sub_22410F0(unsigned __int64 *a1, unsigned __int64 a2, char a3)
{
  unsigned __int64 v3; // r9
  unsigned __int64 *result; // rax

  v3 = a1[1];
  if ( v3 < a2 )
    return sub_2240FD0(a1, a1[1], 0, a2 - v3, a3);
  if ( v3 > a2 )
  {
    result = (unsigned __int64 *)*a1;
    a1[1] = a2;
    *((_BYTE *)result + a2) = 0;
  }
  return result;
}

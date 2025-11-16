// Function: sub_B44850
// Address: 0xb44850
//
__int64 __fastcall sub_B44850(unsigned __int8 *a1, char a2)
{
  unsigned __int64 v2; // rcx
  char v3; // si
  __int64 result; // rax

  v2 = *a1;
  if ( (unsigned __int8)v2 > 0x36u )
  {
    v3 = (2 * a2) & 0x7F | (a1[1] >> 1) & 0x7D;
LABEL_3:
    result = a1[1] & 1;
    a1[1] = result | (2 * v3);
    return result;
  }
  result = 0x40540000000000LL;
  v3 = (2 * a2) & 0x7F | (a1[1] >> 1) & 0x7D;
  if ( !_bittest64(&result, v2) )
    goto LABEL_3;
  a1[1] = a1[1] & 1 | (2 * v3);
  return result;
}

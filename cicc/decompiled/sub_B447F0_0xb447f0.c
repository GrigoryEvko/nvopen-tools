// Function: sub_B447F0
// Address: 0xb447f0
//
__int64 __fastcall sub_B447F0(unsigned __int8 *a1, char a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int8 v3; // al
  char v4; // si
  __int64 result; // rax
  __int64 v6; // rcx

  v2 = *a1;
  v3 = a1[1];
  if ( (unsigned __int8)v2 > 0x36u )
  {
    v4 = (v3 >> 1) & 0x7E | a2 & 0x7F;
LABEL_3:
    result = a1[1] & 1;
    a1[1] = result | (2 * v4);
    return result;
  }
  v6 = 0x40540000000000LL;
  v4 = (v3 >> 1) & 0x7E | a2 & 0x7F;
  if ( !_bittest64(&v6, v2) )
    goto LABEL_3;
  result = a1[1] & 1;
  a1[1] = result | (2 * v4);
  return result;
}

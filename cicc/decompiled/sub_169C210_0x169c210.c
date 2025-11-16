// Function: sub_169C210
// Address: 0x169c210
//
__int64 __fastcall sub_169C210(__int16 **a1, char *a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v6; // rdx

  if ( (unsigned __int8)sub_169B510(a1, (__int64)a2, a3) )
    return 0;
  v6 = a3;
  *((_BYTE *)a1 + 18) = (8 * (*a2 == 45)) | *((_BYTE *)a1 + 18) & 0xF7;
  if ( ((*a2 - 43) & 0xFD) == 0 )
  {
    ++a2;
    v6 = a3 - 1;
  }
  if ( v6 > 1 && *a2 == 48 && (a2[1] & 0xDF) == 0x58 )
    return sub_169A460((__int64)a1, a2 + 2, v6 - 2, a4);
  else
    return sub_169BCA0(a1, a2, v6, a4);
}

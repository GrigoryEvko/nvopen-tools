// Function: sub_22157B0
// Address: 0x22157b0
//
volatile signed __int32 **__fastcall sub_22157B0(
        volatile signed __int32 **a1,
        size_t a2,
        __int64 a3,
        unsigned __int64 a4,
        char a5)
{
  _BYTE *v7; // rbp

  if ( a4 > a3 + 0x3FFFFFFFFFFFFFF9LL - *((_QWORD *)*a1 - 3) )
    sub_4262D8((__int64)"basic_string::_M_replace_aux");
  sub_2215540(a1, a2, a3, a4);
  if ( !a4 )
    return a1;
  v7 = (char *)*a1 + a2;
  if ( a4 != 1 )
  {
    memset(v7, a5, a4);
    return a1;
  }
  *v7 = a5;
  return a1;
}

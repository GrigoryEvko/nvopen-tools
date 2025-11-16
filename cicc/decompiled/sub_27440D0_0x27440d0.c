// Function: sub_27440D0
// Address: 0x27440d0
//
bool __fastcall sub_27440D0(
        __int64 a1,
        unsigned int a2,
        unsigned __int8 *a3,
        _BYTE *a4,
        int a5,
        unsigned int a6,
        __int64 a7)
{
  bool result; // al
  char v11; // [rsp-10h] [rbp-50h]

  sub_2743740(a1, a2, a3, a4, a5, a6, a7, 0);
  result = sub_B52830(a2);
  if ( result )
  {
    sub_2743740(a1, a2, a3, a4, a5, a6, a7, 1);
    return v11;
  }
  return result;
}

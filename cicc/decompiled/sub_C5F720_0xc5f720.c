// Function: sub_C5F720
// Address: 0xc5f720
//
unsigned __int64 __fastcall sub_C5F720(
        _QWORD *a1,
        unsigned __int64 *a2,
        unsigned int a3,
        unsigned __int64 *a4,
        __int64 a5)
{
  if ( a3 == 4 )
    return (unsigned int)sub_C5F610((__int64)a1, a2, a4, (__int64)a4, a5);
  if ( a3 <= 4 )
  {
    if ( a3 == 1 )
      return (unsigned __int8)sub_C5F410(a1, a2, a4, (__int64)a4, a5);
    if ( a3 == 2 )
      return (unsigned __int16)sub_C5F510((__int64)a1, a2, a4, (__int64)a4, a5);
LABEL_10:
    BUG();
  }
  if ( a3 != 8 )
    goto LABEL_10;
  return sub_C5F710((__int64)a1, a2, a4, (__int64)a4, a5);
}

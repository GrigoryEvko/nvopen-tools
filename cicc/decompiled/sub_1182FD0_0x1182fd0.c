// Function: sub_1182FD0
// Address: 0x1182fd0
//
unsigned __int8 *__fastcall sub_1182FD0(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 a8)
{
  if ( *(_QWORD *)(a6 + 8) == *(_QWORD *)(a2 + 8)
    && (a8 == a5 || a8 == a4)
    && a3 == a7
    && (unsigned int)(a3 - 7) > 1
    && a3 )
  {
    return sub_F162A0(a1, a6, a2);
  }
  else
  {
    return 0;
  }
}

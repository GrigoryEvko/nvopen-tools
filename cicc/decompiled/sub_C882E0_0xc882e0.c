// Function: sub_C882E0
// Address: 0xc882e0
//
__int64 __fastcall sub_C882E0(
        _BYTE *a1,
        size_t a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 a8,
        char a9,
        unsigned int a10,
        _QWORD *a11,
        _BYTE *a12,
        int a13,
        char a14)
{
  __int64 v19; // [rsp+38h] [rbp-48h] BYREF

  sub_C86E50(&v19);
  if ( a12 )
  {
    *a12 = 0;
    if ( !(unsigned __int8)sub_C87C00((__pid_t *)&v19, a1, a2, a3, a4, a9, a7, a8, a5, a6, a10, a11, a14) )
      *a12 = 1;
  }
  else
  {
    sub_C87C00((__pid_t *)&v19, a1, a2, a3, a4, a9, a7, a8, a5, a6, a10, a11, a14);
  }
  return v19;
}

// Function: sub_30DF350
// Address: 0x30df350
//
__int64 __fastcall sub_30DF350(
        __int64 a1,
        __int64 a2,
        int *a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v13; // rdx

  v13 = *(_QWORD *)(a2 - 32);
  if ( v13 )
  {
    if ( *(_BYTE *)v13 )
    {
      v13 = 0;
    }
    else if ( *(_QWORD *)(v13 + 24) != *(_QWORD *)(a2 + 80) )
    {
      v13 = 0;
    }
  }
  sub_30DEDC0(a1, (unsigned __int8 *)a2, v13, a3, a4, a11, a5, a6, a7, a8, a9, a10, a12);
  return a1;
}

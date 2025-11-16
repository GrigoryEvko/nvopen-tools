// Function: sub_33E28A0
// Address: 0x33e28a0
//
__int64 __fastcall sub_33E28A0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  int v8; // r13d
  unsigned __int16 v11; // ax

  v8 = a5;
  if ( *(_DWORD *)(a2 + 24) == 51 )
  {
    if ( (unsigned __int8)sub_33E2390(a1, a4, a5, 1) || (unsigned __int8)sub_33E2470(a1, a4) )
      return a4;
    return a7;
  }
  if ( *(_DWORD *)(a4 + 24) == 51 )
    return a7;
  if ( *(_DWORD *)(a7 + 24) == 51 )
    return a4;
  v11 = sub_33E2690(a1, a2, a3, 1u, a5, a6);
  if ( HIBYTE(v11) )
  {
    if ( (_BYTE)v11 )
      return a4;
    return a7;
  }
  if ( a4 == a7 && (_DWORD)a8 == v8 )
    return a4;
  return 0;
}

// Function: sub_34CD310
// Address: 0x34cd310
//
__int64 __fastcall sub_34CD310(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  if ( BYTE4(a8) )
    return sub_BCD140(a2, 8 * (int)a8);
  else
    return sub_BCB2B0(a2);
}

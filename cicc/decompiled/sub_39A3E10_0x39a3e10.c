// Function: sub_39A3E10
// Address: 0x39a3e10
//
__int64 __fastcall sub_39A3E10(__int64 a1, __int64 a2, __int16 a3, __int64 a4, __int64 a5)
{
  unsigned __int16 v7; // ax

  if ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 240LL) + 356LL) )
    return sub_39A3D70(a1, a2, a3, a4, a5);
  v7 = sub_398C0A0(*(_QWORD *)(a1 + 200));
  return sub_39A3990(a1, (__int64 *)(a2 + 8), a3, v7 < 4u ? 6 : 23, a4);
}

// Function: sub_1D5AC20
// Address: 0x1d5ac20
//
__int64 __fastcall sub_1D5AC20(__int64 a1)
{
  _QWORD *v1; // r12
  __int64 v3; // rsi

  v1 = *(_QWORD **)(a1 + 8);
  if ( *(_BYTE *)(a1 + 24) )
  {
    if ( v1[5] )
      sub_15F2070(*(_QWORD **)(a1 + 8));
    return sub_15F2180((__int64)v1, *(_QWORD *)(a1 + 16));
  }
  else
  {
    v3 = sub_157EE30(*(_QWORD *)(a1 + 16));
    if ( v3 )
      v3 -= 24;
    if ( v1[5] )
      return sub_15F22F0(v1, v3);
    else
      return sub_15F2120((__int64)v1, v3);
  }
}

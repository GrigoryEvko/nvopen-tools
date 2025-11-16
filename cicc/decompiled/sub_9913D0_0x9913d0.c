// Function: sub_9913D0
// Address: 0x9913d0
//
bool __fastcall sub_9913D0(__int64 a1, _QWORD *a2, _QWORD *a3, _QWORD *a4)
{
  _BYTE *v5; // rdi
  _BYTE *v7; // rdi
  _QWORD v8[2]; // [rsp+8h] [rbp-18h] BYREF

  v5 = *(_BYTE **)(a1 - 64);
  v8[0] = 0;
  if ( !v5 )
    BUG();
  if ( *v5 == 84 )
  {
    *a2 = v5;
    if ( !(unsigned __int8)sub_990E50((__int64)v5, v8, a3, a4) )
      return 0;
  }
  else
  {
    *a2 = 0;
    v7 = *(_BYTE **)(a1 - 32);
    if ( *v7 != 84 )
      return 0;
    *a2 = v7;
    if ( !(unsigned __int8)sub_990E50((__int64)v7, v8, a3, a4) )
      return 0;
  }
  return v8[0] == a1;
}

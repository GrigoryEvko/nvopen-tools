// Function: sub_3225360
// Address: 0x3225360
//
__int64 __fastcall sub_3225360(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 (__fastcall *v4)(_QWORD); // r9

  if ( *(_BYTE *)(a1 + 3692) )
  {
    if ( (unsigned __int16)sub_3220AA0(a1) <= 4u )
      v4 = (__int64 (__fastcall *)(_QWORD))sub_E0C7F0;
    else
      v4 = (__int64 (__fastcall *)(_QWORD))sub_E0C700;
  }
  else
  {
    v4 = (__int64 (__fastcall *)(_QWORD))sub_E0C510;
  }
  return sub_3224ED0(a1, a2, a3, 3u, 4u, v4);
}

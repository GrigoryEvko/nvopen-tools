// Function: sub_FFE600
// Address: 0xffe600
//
__int64 __fastcall sub_FFE600(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  _BYTE *v3; // rax

  if ( *(_QWORD *)(a2 + 8) == a3 )
    return a2;
  v3 = (_BYTE *)sub_ADB060(a2, a3);
  return sub_97B670(v3, *(_QWORD *)(a1 + 16), 0);
}

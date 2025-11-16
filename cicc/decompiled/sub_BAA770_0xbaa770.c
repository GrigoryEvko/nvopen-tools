// Function: sub_BAA770
// Address: 0xbaa770
//
char __fastcall sub_BAA770(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rdx

  v1 = (_QWORD *)sub_BA91D0(a1, "RtLibUseGOT", 0xBu);
  if ( v1 )
  {
    v2 = v1[17];
    v1 = *(_QWORD **)(v2 + 24);
    if ( *(_DWORD *)(v2 + 32) > 0x40u )
      v1 = (_QWORD *)*v1;
    LOBYTE(v1) = v1 != 0;
  }
  return (char)v1;
}

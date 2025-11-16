// Function: sub_1633DF0
// Address: 0x1633df0
//
char __fastcall sub_1633DF0(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rdx

  v1 = (_QWORD *)sub_16328F0(a1, "RtLibUseGOT", 0xBu);
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

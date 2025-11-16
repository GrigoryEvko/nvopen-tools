// Function: sub_97D330
// Address: 0x97d330
//
__int64 __fastcall sub_97D330(__int64 a1, _BYTE *a2, _BYTE *a3, _BYTE *a4)
{
  _BYTE *v4; // rax

  if ( *a2 > 0x15u || *a3 > 0x15u || *a4 > 0x15u )
    return 0;
  v4 = (_BYTE *)sub_AD5A90(a2, a3, a4, 0);
  return sub_97B670(v4, *(_QWORD *)(a1 + 8), 0);
}

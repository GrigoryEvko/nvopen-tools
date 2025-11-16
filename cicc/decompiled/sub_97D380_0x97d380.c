// Function: sub_97D380
// Address: 0x97d380
//
__int64 __fastcall sub_97D380(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  _BYTE *v3; // rax

  if ( *a2 > 0x15u || *a3 > 0x15u )
    return 0;
  v3 = (_BYTE *)sub_AD5840(a2, a3, 0);
  return sub_97B670(v3, *(_QWORD *)(a1 + 8), 0);
}

// Function: sub_97D2E0
// Address: 0x97d2e0
//
__int64 __fastcall sub_97D2E0(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // rax

  if ( *a2 > 0x15u || *a3 > 0x15u )
    return 0;
  v6 = (_BYTE *)sub_AD5CE0(a2, a3, a4, a5, 0, a6);
  return sub_97B670(v6, *(_QWORD *)(a1 + 8), 0);
}

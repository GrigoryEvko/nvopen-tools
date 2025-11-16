// Function: sub_97D260
// Address: 0x97d260
//
__int64 __fastcall sub_97D260(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v3; // rax

  if ( *(_QWORD *)(a2 + 8) == a3 )
    return a2;
  v3 = (_BYTE *)sub_ADB060(a2, a3);
  return sub_97B670(v3, *(_QWORD *)(a1 + 8), 0);
}

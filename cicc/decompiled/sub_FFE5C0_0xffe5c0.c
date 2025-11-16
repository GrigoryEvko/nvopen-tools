// Function: sub_FFE5C0
// Address: 0xffe5c0
//
__int64 __fastcall sub_FFE5C0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  _BYTE *v3; // rax

  if ( *(_QWORD *)(a2 + 8) == a3 )
    return a2;
  v3 = (_BYTE *)sub_ADAFB0(a2, a3);
  return sub_97B670(v3, *(_QWORD *)(a1 + 16), 0);
}

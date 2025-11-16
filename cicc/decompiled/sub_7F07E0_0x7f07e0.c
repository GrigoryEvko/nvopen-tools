// Function: sub_7F07E0
// Address: 0x7f07e0
//
void __fastcall sub_7F07E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax

  if ( *(_BYTE *)(a1 + 57) == 13 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL) + 56LL);
    *(_BYTE *)(v6 - 8) &= ~8u;
    sub_7F0280(a1, 1);
  }
  else if ( dword_4F077C4 != 2 || dword_4F077BC )
  {
    sub_7D9E50(a1, a2, a3, a4, a5, a6);
  }
}

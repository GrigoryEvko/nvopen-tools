// Function: sub_3259BC0
// Address: 0x3259bc0
//
void __fastcall sub_3259BC0(__int64 a1, __int64 a2)
{
  if ( *(_BYTE *)(a1 + 28) && *(_QWORD *)(a1 + 32) && (*(_BYTE *)(a1 + 26) || *(_BYTE *)(a1 + 24)) )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 176LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
      *(_QWORD *)(a1 + 40),
      0);
    a2 = 0;
    (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 1072LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
      0);
  }
  sub_3259990(a1, a2);
}

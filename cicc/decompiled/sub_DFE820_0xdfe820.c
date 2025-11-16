// Function: sub_DFE820
// Address: 0xdfe820
//
__int64 __fastcall sub_DFE820(__int64 a1)
{
  bool v1; // zf
  void (__fastcall *v2)(__int64, __int64, __int64); // rax

  v1 = *(_BYTE *)(a1 + 216) == 0;
  *(_QWORD *)a1 = &unk_49DEC00;
  if ( !v1 )
  {
    *(_BYTE *)(a1 + 216) = 0;
    sub_DFE7B0((_QWORD *)(a1 + 208));
  }
  v2 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 192);
  if ( v2 )
    v2(a1 + 176, a1 + 176, 3);
  return sub_BB9280(a1);
}

// Function: sub_2FF0CC0
// Address: 0x2ff0cc0
//
void __fastcall sub_2FF0CC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  void (__fastcall *v3)(__int64, _QWORD *, _QWORD); // rbx
  _QWORD *v4; // rax

  if ( (_DWORD)qword_5028BC8 == 1 )
  {
    v2 = *(_QWORD *)(a1 + 176);
    v3 = *(void (__fastcall **)(__int64, _QWORD *, _QWORD))(*(_QWORD *)v2 + 16LL);
    v4 = sub_2EF8D50(a2);
    v3(v2, v4, 0);
  }
}

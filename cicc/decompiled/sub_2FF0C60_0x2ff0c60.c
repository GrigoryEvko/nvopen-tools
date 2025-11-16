// Function: sub_2FF0C60
// Address: 0x2ff0c60
//
void __fastcall sub_2FF0C60(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  void (__fastcall *v3)(__int64, _QWORD *, _QWORD); // rbx
  __int64 v4; // rax
  _QWORD *v5; // rax

  if ( (_BYTE)qword_5028348 )
  {
    v2 = *(_QWORD *)(a1 + 176);
    v3 = *(void (__fastcall **)(__int64, _QWORD *, _QWORD))(*(_QWORD *)v2 + 16LL);
    v4 = sub_C5F790(a1, a2);
    v5 = sub_2E85230(v4, a2);
    v3(v2, v5, 0);
  }
}

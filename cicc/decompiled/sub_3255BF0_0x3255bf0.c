// Function: sub_3255BF0
// Address: 0x3255bf0
//
void __fastcall sub_3255BF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  void (__fastcall *v6)(__int64, _QWORD, _QWORD); // rbx
  __int64 v7; // rax
  __int64 v8; // r14
  void (__fastcall *v9)(__int64, _QWORD, _QWORD); // rbx
  __int64 v10; // rax

  v4 = *(_QWORD *)(a4 + 224);
  v6 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v4 + 176LL);
  v7 = sub_31DA6B0(a4);
  v6(v4, *(_QWORD *)(v7 + 24), 0);
  sub_3255900(a2, a4, "code_begin");
  v8 = *(_QWORD *)(a4 + 224);
  v9 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v8 + 176LL);
  v10 = sub_31DA6B0(a4);
  v9(v8, *(_QWORD *)(v10 + 32), 0);
  sub_3255900(a2, a4, "data_begin");
}

// Function: sub_3214CC0
// Address: 0x3214cc0
//
__int64 __fastcall sub_3214CC0(__int64 a1, __int64 a2)
{
  void (__fastcall *v2)(__int64, _QWORD, const char *, _QWORD); // rbx
  const char *v3; // rax
  void (__fastcall *v4)(__int64, _QWORD, const char *, _QWORD); // rbx
  const char *v5; // rax
  unsigned __int16 *v6; // rbx
  unsigned __int16 *i; // r13
  void (__fastcall *v8)(__int64, _QWORD, const char *, _QWORD); // r14
  const char *v9; // rax
  void (__fastcall *v10)(__int64, _QWORD, const char *, _QWORD); // r14
  const char *v11; // rax

  v2 = *(void (__fastcall **)(__int64, _QWORD, const char *, _QWORD))(*(_QWORD *)a2 + 424LL);
  v3 = sub_E02B90(*(unsigned __int16 *)(a1 + 12));
  v2(a2, *(unsigned __int16 *)(a1 + 12), v3, 0);
  v4 = *(void (__fastcall **)(__int64, _QWORD, const char *, _QWORD))(*(_QWORD *)a2 + 424LL);
  v5 = sub_E05870(*(unsigned __int8 *)(a1 + 14));
  v4(a2, *(unsigned __int8 *)(a1 + 14), v5, 0);
  v6 = *(unsigned __int16 **)(a1 + 16);
  for ( i = &v6[8 * *(unsigned int *)(a1 + 24)]; i != v6; v6 += 8 )
  {
    v8 = *(void (__fastcall **)(__int64, _QWORD, const char *, _QWORD))(*(_QWORD *)a2 + 424LL);
    v9 = sub_E058A0(*v6);
    v8(a2, *v6, v9, 0);
    v10 = *(void (__fastcall **)(__int64, _QWORD, const char *, _QWORD))(*(_QWORD *)a2 + 424LL);
    v11 = sub_E06AB0(v6[1]);
    v10(a2, v6[1], v11, 0);
    if ( v6[1] == 33 )
      (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)a2 + 416LL))(a2, *((_QWORD *)v6 + 1), 0);
  }
  (*(void (__fastcall **)(__int64, _QWORD, const char *, _QWORD))(*(_QWORD *)a2 + 424LL))(a2, 0, "EOM(1)", 0);
  return (*(__int64 (__fastcall **)(__int64, _QWORD, const char *, _QWORD))(*(_QWORD *)a2 + 424LL))(a2, 0, "EOM(2)", 0);
}

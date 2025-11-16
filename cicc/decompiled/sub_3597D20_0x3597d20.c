// Function: sub_3597D20
// Address: 0x3597d20
//
__int64 __fastcall sub_3597D20(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 (__fastcall *v8)(__int64, const char *); // rbx
  const char *v9; // rax

  sub_2F5F560((__int64)a1, (__int64)a2, a3, a4);
  *a1 = &unk_4A39CE8;
  sub_2F5F560((__int64)(a1 + 9), (__int64)a2, a3, a4);
  a1[18] = a5;
  a1[9] = &unk_4A2B260;
  v8 = *(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)a5 + 16LL);
  v9 = sub_2E791E0(a2);
  return v8(a5, v9);
}

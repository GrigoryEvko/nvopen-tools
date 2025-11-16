// Function: sub_A55520
// Address: 0xa55520
//
__int64 (__fastcall *__fastcall sub_A55520(_QWORD *a1, __int64 a2))(_QWORD *, _QWORD *, __int64)
{
  void (__fastcall *v2)(__int64, __int64, __int64); // rax
  __int64 (__fastcall *result)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 (__fastcall *v6)(__int64, __int64); // rax

  *a1 = &unk_49D9A00;
  v2 = (void (__fastcall *)(__int64, __int64, __int64))a1[12];
  if ( v2 )
  {
    a2 = (__int64)(a1 + 10);
    v2(a2, a2, 3);
  }
  result = (__int64 (__fastcall *)(_QWORD *, _QWORD *, __int64))a1[8];
  if ( result )
  {
    a2 = (__int64)(a1 + 6);
    result = (__int64 (__fastcall *)(_QWORD *, _QWORD *, __int64))result(a1 + 6, a1 + 6, 3);
  }
  v4 = a1[1];
  if ( v4 )
  {
    v5 = a1[1];
    v6 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v4 + 8LL);
    if ( v6 == sub_A554F0 )
    {
      sub_A552A0(v5, a2);
      return (__int64 (__fastcall *)(_QWORD *, _QWORD *, __int64))j_j___libc_free_0(v4, 400);
    }
    else
    {
      return (__int64 (__fastcall *)(_QWORD *, _QWORD *, __int64))((__int64 (__fastcall *)(__int64))v6)(v5);
    }
  }
  return result;
}

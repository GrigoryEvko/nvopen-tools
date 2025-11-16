// Function: sub_2221730
// Address: 0x2221730
//
void __fastcall sub_2221730(__int64 a1, __int64 *a2)
{
  void (__fastcall *v3)(__int64 *); // rax
  const wchar_t *v4; // rsi
  char *v5; // rdi
  _QWORD v6[2]; // [rsp+0h] [rbp-38h] BYREF
  char v7; // [rsp+10h] [rbp-28h] BYREF

  (*(void (__fastcall **)(_QWORD *, __int64))(*(_QWORD *)a1 + 24LL))(v6, a1);
  v3 = (void (__fastcall *)(__int64 *))a2[4];
  if ( v3 )
    v3(a2);
  v4 = (const wchar_t *)v6[0];
  *a2 = (__int64)(a2 + 2);
  sub_2220100(a2, v4, (__int64)&v4[v6[1]]);
  v5 = (char *)v6[0];
  a2[4] = (__int64)sub_221F8F0;
  if ( v5 != &v7 )
    j___libc_free_0((unsigned __int64)v5);
}

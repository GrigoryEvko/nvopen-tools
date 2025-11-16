// Function: sub_E21CC0
// Address: 0xe21cc0
//
__int64 __fastcall sub_E21CC0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  size_t v3; // rsi
  const void *v4; // rdx
  const void *v6; // [rsp+0h] [rbp-40h] BYREF
  size_t v7; // [rsp+8h] [rbp-38h]
  __int64 v8; // [rsp+10h] [rbp-30h]
  __int64 v9; // [rsp+18h] [rbp-28h]
  int v10; // [rsp+20h] [rbp-20h]

  v2 = *a2;
  v6 = 0;
  v7 = 0;
  v8 = 0;
  v9 = -1;
  v10 = 1;
  (*(void (__fastcall **)(__int64 *, const void **, _QWORD))(v2 + 16))(a2, &v6, 0);
  v3 = sub_E213F0(a1, v7, v6);
  sub_E21AF0(a1, v3, v4);
  return _libc_free(v6, v3);
}

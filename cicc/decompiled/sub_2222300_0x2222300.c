// Function: sub_2222300
// Address: 0x2222300
//
void __fastcall sub_2222300(
        __int64 a1,
        __int64 *a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        const wchar_t *a6,
        __int64 a7)
{
  void (__fastcall *v11)(__int64 *); // rax
  const wchar_t *v12; // rsi
  char *v13; // rdi
  __int64 v14[2]; // [rsp+10h] [rbp-78h] BYREF
  _BYTE v15[16]; // [rsp+20h] [rbp-68h] BYREF
  _QWORD v16[2]; // [rsp+30h] [rbp-58h] BYREF
  char v17; // [rsp+40h] [rbp-48h] BYREF

  v14[0] = (__int64)v15;
  sub_221FEA0(v14, a6, (__int64)&a6[a7]);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD, _QWORD, __int64 *))(*(_QWORD *)a1 + 24LL))(
    v16,
    a1,
    a3,
    a4,
    a5,
    v14);
  v11 = (void (__fastcall *)(__int64 *))a2[4];
  if ( v11 )
    v11(a2);
  v12 = (const wchar_t *)v16[0];
  *a2 = (__int64)(a2 + 2);
  sub_2220100(a2, v12, (__int64)&v12[v16[1]]);
  v13 = (char *)v16[0];
  a2[4] = (__int64)sub_221F8F0;
  if ( v13 != &v17 )
    j___libc_free_0((unsigned __int64)v13);
  if ( (_BYTE *)v14[0] != v15 )
    j___libc_free_0(v14[0]);
}

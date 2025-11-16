// Function: sub_23CF390
// Address: 0x23cf390
//
__int64 __fastcall sub_23CF390(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  _QWORD *v3; // rbx
  __int64 (*v4)(); // rax
  __int64 v5; // rcx
  __int64 v6; // rdi
  __int64 result; // rax
  __int64 v8; // [rsp-100h] [rbp-100h]
  const char *v9[4]; // [rsp-F8h] [rbp-F8h] BYREF
  __int16 v10; // [rsp-D8h] [rbp-D8h]
  const char *v11; // [rsp-C8h] [rbp-C8h] BYREF
  const char *v12; // [rsp-C0h] [rbp-C0h]
  __int64 v13; // [rsp-B8h] [rbp-B8h]
  _BYTE v14[176]; // [rsp-B0h] [rbp-B0h] BYREF

  v2 = *(__int64 (**)())(*(_QWORD *)a1 + 24LL);
  if ( v2 == sub_23CE280 )
    BUG();
  v3 = (_QWORD *)v2();
  v4 = *(__int64 (**)())(*v3 + 248LL);
  if ( v4 == sub_23CE400 || (result = ((__int64 (__fastcall *)(_QWORD *, __int64, __int64))v4)(v3, a2, a1)) == 0 )
  {
    v5 = v3[116];
    v12 = 0;
    v11 = v14;
    v13 = 128;
    sub_23CF320(a1, (__int64)&v11, a2, v5, 0);
    v6 = v3[115];
    v10 = 261;
    v9[0] = v11;
    v9[1] = v12;
    result = sub_E6C460(v6, v9);
    if ( v11 != v14 )
    {
      v8 = result;
      _libc_free((unsigned __int64)v11);
      return v8;
    }
  }
  return result;
}

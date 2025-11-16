// Function: sub_B865A0
// Address: 0xb865a0
//
_BYTE *__fastcall sub_B865A0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  _BYTE *result; // rax
  _QWORD v4[2]; // [rsp-E8h] [rbp-E8h] BYREF
  _BYTE v5[64]; // [rsp-D8h] [rbp-D8h] BYREF
  _BYTE *v6; // [rsp-98h] [rbp-98h]
  __int64 v7; // [rsp-90h] [rbp-90h]
  _BYTE v8[16]; // [rsp-88h] [rbp-88h] BYREF
  _QWORD v9[2]; // [rsp-78h] [rbp-78h] BYREF
  _BYTE v10[16]; // [rsp-68h] [rbp-68h] BYREF
  _BYTE *v11; // [rsp-58h] [rbp-58h]
  __int64 v12; // [rsp-50h] [rbp-50h]
  _BYTE v13[72]; // [rsp-48h] [rbp-48h] BYREF

  if ( (int)qword_4F81B88 > 3 )
  {
    v2 = *a2;
    v4[1] = 0x800000000LL;
    v7 = 0x200000000LL;
    v9[1] = 0x200000000LL;
    v11 = v13;
    v4[0] = v5;
    v6 = v8;
    v9[0] = v10;
    v12 = 0;
    v13[0] = 0;
    (*(void (__fastcall **)(__int64 *, _QWORD *))(v2 + 88))(a2, v4);
    sub_B86160(a1, "Preserved", 9u, (__int64)a2, (__int64)v9);
    result = v13;
    if ( v11 != v13 )
      result = (_BYTE *)_libc_free(v11, "Preserved");
    if ( (_BYTE *)v9[0] != v10 )
      result = (_BYTE *)_libc_free(v9[0], "Preserved");
    if ( v6 != v8 )
      result = (_BYTE *)_libc_free(v6, "Preserved");
    if ( (_BYTE *)v4[0] != v5 )
      return (_BYTE *)_libc_free(v4[0], "Preserved");
  }
  return result;
}

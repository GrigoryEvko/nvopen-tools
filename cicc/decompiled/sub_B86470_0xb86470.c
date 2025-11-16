// Function: sub_B86470
// Address: 0xb86470
//
_BYTE *__fastcall sub_B86470(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  _BYTE *result; // rax
  _QWORD v4[2]; // [rsp-E8h] [rbp-E8h] BYREF
  _BYTE v5[64]; // [rsp-D8h] [rbp-D8h] BYREF
  _BYTE *v6; // [rsp-98h] [rbp-98h]
  __int64 v7; // [rsp-90h] [rbp-90h]
  _BYTE v8[16]; // [rsp-88h] [rbp-88h] BYREF
  _BYTE *v9; // [rsp-78h] [rbp-78h]
  __int64 v10; // [rsp-70h] [rbp-70h]
  _BYTE v11[16]; // [rsp-68h] [rbp-68h] BYREF
  _BYTE *v12; // [rsp-58h] [rbp-58h]
  __int64 v13; // [rsp-50h] [rbp-50h]
  _BYTE v14[72]; // [rsp-48h] [rbp-48h] BYREF

  if ( (int)qword_4F81B88 > 3 )
  {
    v2 = *a2;
    v4[1] = 0x800000000LL;
    v7 = 0x200000000LL;
    v10 = 0x200000000LL;
    v12 = v14;
    v4[0] = v5;
    v6 = v8;
    v9 = v11;
    v13 = 0;
    v14[0] = 0;
    (*(void (__fastcall **)(__int64 *, _QWORD *))(v2 + 88))(a2, v4);
    sub_B86160(a1, "Required", 8u, (__int64)a2, (__int64)v4);
    result = v14;
    if ( v12 != v14 )
      result = (_BYTE *)_libc_free(v12, "Required");
    if ( v9 != v11 )
      result = (_BYTE *)_libc_free(v9, "Required");
    if ( v6 != v8 )
      result = (_BYTE *)_libc_free(v6, "Required");
    if ( (_BYTE *)v4[0] != v5 )
      return (_BYTE *)_libc_free(v4[0], "Required");
  }
  return result;
}

// Function: sub_1615D60
// Address: 0x1615d60
//
void __fastcall sub_1615D60(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3[2]; // [rsp-E8h] [rbp-E8h] BYREF
  _BYTE v4[64]; // [rsp-D8h] [rbp-D8h] BYREF
  _BYTE *v5; // [rsp-98h] [rbp-98h]
  __int64 v6; // [rsp-90h] [rbp-90h]
  _BYTE v7[16]; // [rsp-88h] [rbp-88h] BYREF
  _BYTE *v8; // [rsp-78h] [rbp-78h]
  __int64 v9; // [rsp-70h] [rbp-70h]
  _BYTE v10[16]; // [rsp-68h] [rbp-68h] BYREF
  _BYTE *v11; // [rsp-58h] [rbp-58h]
  __int64 v12; // [rsp-50h] [rbp-50h]
  _BYTE v13[72]; // [rsp-48h] [rbp-48h] BYREF

  if ( dword_4F9EB40 > 3 )
  {
    v2 = *a2;
    v3[1] = 0x800000000LL;
    v6 = 0x200000000LL;
    v9 = 0x200000000LL;
    v11 = v13;
    v3[0] = (unsigned __int64)v4;
    v5 = v7;
    v8 = v10;
    v12 = 0;
    v13[0] = 0;
    (*(void (__fastcall **)(__int64 *, unsigned __int64 *))(v2 + 88))(a2, v3);
    sub_1615A50(a1, "Required", 8u, (__int64)a2, (__int64)v3);
    if ( v11 != v13 )
      _libc_free((unsigned __int64)v11);
    if ( v8 != v10 )
      _libc_free((unsigned __int64)v8);
    if ( v5 != v7 )
      _libc_free((unsigned __int64)v5);
    if ( (_BYTE *)v3[0] != v4 )
      _libc_free(v3[0]);
  }
}

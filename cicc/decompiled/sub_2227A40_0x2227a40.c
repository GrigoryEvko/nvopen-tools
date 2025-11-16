// Function: sub_2227A40
// Address: 0x2227a40
//
__int64 __fastcall sub_2227A40(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, unsigned int a6, long double a7)
{
  __int64 v7; // rsi
  int v8; // eax
  int v9; // r12d
  void *v10; // rsp
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v15; // [rsp-1Eh] [rbp-100h]
  __int64 v16; // [rsp-16h] [rbp-F8h]
  char v17[15]; // [rsp-Eh] [rbp-F0h] BYREF
  __int64 v18; // [rsp+4Ah] [rbp-98h]
  unsigned int v19; // [rsp+52h] [rbp-90h]
  int v20; // [rsp+56h] [rbp-8Ch]
  __int64 v21; // [rsp+5Ah] [rbp-88h]
  __int64 v22; // [rsp+62h] [rbp-80h]
  __int64 v23; // [rsp+6Ah] [rbp-78h]
  __int64 v24; // [rsp+72h] [rbp-70h]
  _BYTE *v25; // [rsp+7Ah] [rbp-68h]
  volatile signed __int32 *v26; // [rsp+8Ah] [rbp-58h] BYREF
  unsigned __int64 v27[2]; // [rsp+92h] [rbp-50h] BYREF
  _BYTE v28[64]; // [rsp+A2h] [rbp-40h] BYREF

  v23 = a1;
  v22 = a2;
  v7 = a5 + 208;
  v21 = a3;
  v20 = a4;
  v18 = a5;
  v19 = a6;
  sub_2208E20(&v26, (volatile signed __int32 **)(a5 + 208));
  v24 = sub_2243120(&v26);
  v27[0] = sub_2208E60(&v26, v7);
  v8 = sub_2218500((__int64)v27, v17, 64, "%.*Lf", 0, a7);
  if ( v8 > 63 )
  {
    v9 = v8 + 1;
    v10 = alloca(v8 + 1 + 8LL);
    v27[0] = sub_2208E60(v16, v15);
    v8 = sub_2218500((__int64)v27, v17, v9, "%.*Lf", 0, a7);
  }
  v11 = v8;
  v25 = v28;
  v27[0] = (unsigned __int64)v28;
  sub_2251800(v27, v8, 0);
  (*(void (__fastcall **)(__int64, char *, char *, unsigned __int64))(*(_QWORD *)v24 + 88LL))(
    v24,
    v17,
    &v17[v11],
    v27[0]);
  if ( (_BYTE)v20 )
    v12 = sub_22269C0(v23, v22, v21, v18, v19, v27);
  else
    v12 = sub_2227200(v23, v22, v21, v18, v19, v27);
  v13 = v12;
  if ( (_BYTE *)v27[0] != v25 )
    j___libc_free_0(v27[0]);
  sub_2209150(&v26);
  return v13;
}

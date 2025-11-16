// Function: sub_221BFD0
// Address: 0x221bfd0
//
__int64 __fastcall sub_221BFD0(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, int a6, long double a7)
{
  __int64 v7; // rsi
  _BYTE *v8; // r15
  int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // r8
  int v12; // r12d
  void *v13; // rsp
  size_t v14; // r12
  char v15; // al
  _BYTE *(__fastcall *v16)(__int64, char *, char *, void *); // rax
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v20; // [rsp-1Eh] [rbp-100h]
  __int64 v21; // [rsp-16h] [rbp-F8h]
  __int64 v22; // [rsp-16h] [rbp-F8h]
  char v23[15]; // [rsp-Eh] [rbp-F0h] BYREF
  void *dest; // [rsp+42h] [rbp-A0h]
  char *v25; // [rsp+4Ah] [rbp-98h]
  __int64 v26; // [rsp+52h] [rbp-90h]
  int v27; // [rsp+5Ah] [rbp-88h]
  int v28; // [rsp+5Eh] [rbp-84h]
  __int64 v29; // [rsp+62h] [rbp-80h]
  __int64 v30; // [rsp+6Ah] [rbp-78h]
  __int64 v31; // [rsp+72h] [rbp-70h]
  char *v32; // [rsp+7Ah] [rbp-68h]
  volatile signed __int32 *v33; // [rsp+8Ah] [rbp-58h] BYREF
  unsigned __int64 v34[2]; // [rsp+92h] [rbp-50h] BYREF
  _BYTE v35[64]; // [rsp+A2h] [rbp-40h] BYREF

  v31 = a1;
  v30 = a2;
  v7 = a5 + 208;
  v29 = a3;
  v28 = a4;
  v26 = a5;
  v27 = a6;
  sub_2208E20(&v33, (volatile signed __int32 **)(a5 + 208));
  v8 = (_BYTE *)sub_222F790(&v33);
  v34[0] = sub_2208E60(&v33, v7);
  v9 = sub_2218500((__int64)v34, v23, 64, "%.*Lf", 0, a7);
  if ( v9 > 63 )
  {
    v12 = v9 + 1;
    v13 = alloca(v9 + 1 + 8LL);
    v32 = v23;
    v34[0] = sub_2208E60(v21, v20);
    v9 = sub_2218500((__int64)v34, v32, v12, "%.*Lf", 0, a7);
    v10 = v22;
  }
  v14 = v9;
  v32 = v35;
  v34[0] = (unsigned __int64)v35;
  sub_2240A50(v34, v9, 0, v10, v11);
  dest = (void *)v34[0];
  v25 = &v23[v14];
  v15 = v8[56];
  if ( v15 == 1 )
    goto LABEL_6;
  if ( v15 )
  {
    v16 = *(_BYTE *(__fastcall **)(__int64, char *, char *, void *))(*(_QWORD *)v8 + 56LL);
    if ( v16 == sub_2216D40 )
      goto LABEL_6;
LABEL_15:
    v16((__int64)v8, v23, v25, dest);
    goto LABEL_8;
  }
  sub_2216D60((__int64)v8);
  v16 = *(_BYTE *(__fastcall **)(__int64, char *, char *, void *))(*(_QWORD *)v8 + 56LL);
  if ( v16 != sub_2216D40 )
    goto LABEL_15;
LABEL_6:
  if ( v25 != v23 )
    memcpy(dest, v23, v14);
LABEL_8:
  if ( (_BYTE)v28 )
    v17 = sub_221AF70(v31, v30, v29, v26, v27, (__int64)v34);
  else
    v17 = sub_221B7A0(v31, v30, v29, v26, v27, (__int64)v34);
  v18 = v17;
  if ( (char *)v34[0] != v32 )
    j___libc_free_0(v34[0]);
  sub_2209150(&v33);
  return v18;
}

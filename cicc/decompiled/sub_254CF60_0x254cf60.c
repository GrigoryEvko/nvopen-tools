// Function: sub_254CF60
// Address: 0x254cf60
//
__int64 __fastcall sub_254CF60(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  unsigned __int8 *v3; // rax
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // r9
  bool v7; // zf
  _QWORD *v8; // r13
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  char v13; // dl
  __int64 v14; // rbx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned int v19; // r13d
  int v20; // eax
  unsigned int v21; // r12d
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  _QWORD *v25; // rax
  __int64 v26; // [rsp+8h] [rbp-F8h]
  _BYTE *v27; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v28; // [rsp+28h] [rbp-D8h]
  _BYTE v29[16]; // [rsp+30h] [rbp-D0h] BYREF
  _BYTE *v30; // [rsp+40h] [rbp-C0h] BYREF
  size_t v31; // [rsp+48h] [rbp-B8h]
  _BYTE v32[16]; // [rsp+50h] [rbp-B0h] BYREF
  _QWORD *v33; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v34; // [rsp+68h] [rbp-98h]
  _QWORD v35[4]; // [rsp+70h] [rbp-90h] BYREF
  void *v36; // [rsp+90h] [rbp-70h] BYREF
  __int64 v37; // [rsp+98h] [rbp-68h]
  __int64 v38; // [rsp+A0h] [rbp-60h]
  __int64 v39; // [rsp+A8h] [rbp-58h]
  __int64 v40; // [rsp+B0h] [rbp-50h]
  __int64 v41; // [rsp+B8h] [rbp-48h]
  const void **v42; // [rsp+C0h] [rbp-40h]

  v2 = (__int64 *)(a1 + 72);
  v3 = sub_250CBE0((__int64 *)(a1 + 72), a2);
  v4 = sub_B2BE50((__int64)v3);
  v7 = *(_BYTE *)(a1 + 96) == 0;
  v8 = (_QWORD *)v4;
  v27 = v29;
  v28 = 0x200000000LL;
  v33 = v35;
  v34 = 0x200000000LL;
  if ( !v7 || (v13 = *(_BYTE *)(a1 + 97)) != 0 )
  {
    v30 = v32;
    v41 = 0x100000000LL;
    v32[0] = 0;
    v36 = &unk_49DD210;
    v31 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    v42 = (const void **)&v30;
    sub_CB5980((__int64)&v36, 0, 0, 0);
    sub_254C2A0((char *)(a1 + 96), (__int64)&v36);
    v36 = &unk_49DD210;
    sub_CB5840((__int64)&v36);
    v9 = sub_A78730(v8, "denormal-fp-math", 0x10u, v30, v31);
    v11 = (unsigned int)v28;
    v12 = (unsigned int)v28 + 1LL;
    if ( v12 > HIDWORD(v28) )
    {
      v26 = v9;
      sub_C8D5F0((__int64)&v27, v29, v12, 8u, v9, v10);
      v11 = (unsigned int)v28;
      v9 = v26;
    }
    *(_QWORD *)&v27[8 * v11] = v9;
    LODWORD(v28) = v28 + 1;
    sub_2240A30((unsigned __int64 *)&v30);
    v13 = *(_BYTE *)(a1 + 97);
    if ( *(_BYTE *)(a1 + 98) != *(_BYTE *)(a1 + 96) )
      goto LABEL_5;
  }
  else
  {
    v35[1] = 16;
    v35[0] = "denormal-fp-math";
    LODWORD(v34) = 1;
    if ( *(_BYTE *)(a1 + 98) )
      goto LABEL_5;
  }
  if ( *(_BYTE *)(a1 + 99) == v13 )
  {
    v23 = (unsigned int)v34;
    v24 = (unsigned int)v34 + 1LL;
    if ( v24 > HIDWORD(v34) )
    {
      sub_C8D5F0((__int64)&v33, v35, v24, 0x10u, v5, v6);
      v23 = (unsigned int)v34;
    }
    v25 = &v33[2 * v23];
    *v25 = "denormal-fp-math-f32";
    v25[1] = 20;
    LODWORD(v34) = v34 + 1;
    goto LABEL_8;
  }
LABEL_5:
  v30 = v32;
  v41 = 0x100000000LL;
  v32[0] = 0;
  v36 = &unk_49DD210;
  v31 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v42 = (const void **)&v30;
  sub_CB5980((__int64)&v36, 0, 0, 0);
  sub_254C2A0((char *)(a1 + 98), (__int64)&v36);
  v36 = &unk_49DD210;
  sub_CB5840((__int64)&v36);
  v14 = sub_A78730(v8, "denormal-fp-math-f32", 0x14u, v30, v31);
  v17 = (unsigned int)v28;
  v18 = (unsigned int)v28 + 1LL;
  if ( v18 > HIDWORD(v28) )
  {
    sub_C8D5F0((__int64)&v27, v29, v18, 8u, v15, v16);
    v17 = (unsigned int)v28;
  }
  *(_QWORD *)&v27[8 * v17] = v14;
  LODWORD(v28) = v28 + 1;
  sub_2240A30((unsigned __int64 *)&v30);
LABEL_8:
  v19 = sub_2516380(a2, v2, (__int64)v27, (unsigned int)v28, 1);
  v20 = sub_2515790(a2, v2, (__int64)v33, (unsigned int)v34);
  v21 = sub_250C0B0(v20, v19);
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  return v21;
}

// Function: sub_C5EA20
// Address: 0xc5ea20
//
__int64 __fastcall sub_C5EA20(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned __int64 *a4, __int64 a5)
{
  unsigned __int64 v6; // rbx
  unsigned __int64 v8; // r14
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // [rsp+0h] [rbp-E0h]
  __int64 v13; // [rsp+8h] [rbp-D8h]
  _QWORD *v14; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v15; // [rsp+28h] [rbp-B8h]
  _QWORD v16[2]; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int64 v17; // [rsp+40h] [rbp-A0h] BYREF
  const char *v18; // [rsp+48h] [rbp-98h]
  unsigned __int64 v19; // [rsp+50h] [rbp-90h]
  unsigned __int64 v20; // [rsp+58h] [rbp-88h]
  unsigned __int64 v21; // [rsp+60h] [rbp-80h]
  unsigned __int64 v22; // [rsp+70h] [rbp-70h] BYREF
  __int64 v23; // [rsp+78h] [rbp-68h]
  __int64 v24; // [rsp+80h] [rbp-60h]
  __int64 v25; // [rsp+88h] [rbp-58h]
  __int64 v26; // [rsp+90h] [rbp-50h]
  __int64 v27; // [rsp+98h] [rbp-48h]
  _QWORD *v28; // [rsp+A0h] [rbp-40h]

  v6 = a2 + a3;
  if ( !__CFADD__(a2, a3) && *(_QWORD *)(a1 + 8) > v6 - 1 )
    return 1;
  if ( !a4 )
    return 0;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = sub_2241E50(a1, a2, a3, a4, a5);
  if ( v8 >= a2 )
  {
    v12 = v9;
    v27 = 0x100000000LL;
    v14 = v16;
    v28 = &v14;
    v22 = (unsigned __int64)&unk_49DD210;
    v15 = 0;
    LOBYTE(v16[0]) = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    sub_CB5980(&v22, 0, 0, 0);
    v20 = a2;
    v18 = "unexpected end of data at offset 0x%zx while reading [0x%lx, 0x%lx)";
    v19 = v6;
    v21 = v8;
    v17 = (unsigned __int64)&unk_49DC5A0;
    sub_CB6620(&v22, &v17);
    v22 = (unsigned __int64)&unk_49DD210;
    sub_CB5840(&v22);
    sub_C5E9B0((__int64 *)&v17, (__int64)&v14, 0x54u, v12);
    if ( v14 != v16 )
      j_j___libc_free_0(v14, v16[0] + 1LL);
    v10 = *a4;
    if ( (*a4 & 1) == 0 && (v10 & 0xFFFFFFFFFFFFFFFELL) == 0 )
    {
      *a4 = v17 | v10 | 1;
      return 0;
    }
LABEL_17:
    sub_C63C30(a4);
  }
  v13 = v9;
  v27 = 0x100000000LL;
  v28 = &v14;
  v22 = (unsigned __int64)&unk_49DD210;
  v14 = v16;
  v15 = 0;
  LOBYTE(v16[0]) = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  sub_CB5980(&v22, 0, 0, 0);
  v18 = "offset 0x%lx is beyond the end of data at 0x%zx";
  v19 = v8;
  v20 = a2;
  v17 = (unsigned __int64)&unk_49D98C0;
  sub_CB6620(&v22, &v17);
  v22 = (unsigned __int64)&unk_49DD210;
  sub_CB5840(&v22);
  sub_C5E9B0((__int64 *)&v22, (__int64)&v14, 0x16u, v13);
  if ( v14 != v16 )
    j_j___libc_free_0(v14, v16[0] + 1LL);
  v11 = *a4;
  if ( (*a4 & 1) != 0 || (v11 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_17;
  *a4 = v22 | v11 | 1;
  return 0;
}

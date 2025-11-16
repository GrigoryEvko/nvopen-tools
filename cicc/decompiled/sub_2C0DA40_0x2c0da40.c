// Function: sub_2C0DA40
// Address: 0x2c0da40
//
unsigned __int64 __fastcall sub_2C0DA40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v6; // r15
  __int64 *v7; // r14
  __int64 v8; // r13
  __int64 *v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  bool v15; // of
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // r12
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r15
  __int128 v23; // [rsp-18h] [rbp-138h]
  __int64 v24; // [rsp+0h] [rbp-120h]
  __int64 *v25; // [rsp+10h] [rbp-110h]
  __int64 v26; // [rsp+10h] [rbp-110h]
  __int64 v27; // [rsp+18h] [rbp-108h]
  _QWORD v28[4]; // [rsp+30h] [rbp-F0h] BYREF
  _BYTE v29[24]; // [rsp+50h] [rbp-D0h] BYREF
  char *v30; // [rsp+68h] [rbp-B8h]
  char v31; // [rsp+78h] [rbp-A8h] BYREF
  char *v32; // [rsp+98h] [rbp-88h]
  char v33; // [rsp+A8h] [rbp-78h] BYREF

  v3 = a3 + 16;
  v25 = (__int64 *)sub_2BFD6A0(a3 + 16, **(_QWORD **)(a1 + 48));
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL);
  v7 = (__int64 *)sub_2BFD6A0(v3, v6);
  v8 = sub_BCE1B0(v7, a2);
  v24 = sub_DFD800(*(_QWORD *)a3, 0x11u, v8, *(_DWORD *)(a3 + 176), 0, 0, 0, 0, 0, 0);
  v27 = v24;
  if ( !sub_2BF04A0(v6) )
  {
    v20 = *(_QWORD *)(v6 + 40);
    if ( *(_BYTE *)v20 == 17 )
    {
      if ( *(_DWORD *)(v20 + 32) <= 0x40u )
        v21 = *(_QWORD *)(v20 + 24);
      else
        v21 = **(_QWORD **)(v20 + 24);
      v22 = 0;
      if ( v21 != 1 )
        v22 = v24;
      v27 = v22;
    }
  }
  v26 = sub_BCE1B0(v25, a2);
  v9 = (__int64 *)sub_BCB2A0(*(_QWORD **)(a3 + 64));
  v10 = sub_BCE1B0(v9, a2);
  v11 = *(_QWORD **)(a3 + 64);
  v28[2] = v10;
  v28[0] = v26;
  v28[1] = v7;
  v12 = sub_BCB120(v11);
  *((_QWORD *)&v23 + 1) = 1;
  *(_QWORD *)&v23 = 0;
  sub_DF8CB0((__int64)v29, 161, v12, (char *)v28, 3, 0, 0, v23);
  v13 = sub_DFD800(*(_QWORD *)a3, *(_DWORD *)(a1 + 96), v8, *(_DWORD *)(a3 + 176), 0, 0, 0, 0, 0, 0);
  v14 = sub_DFD690(*(_QWORD *)a3, (__int64)v29);
  v15 = __OFADD__(v27, v14);
  v16 = v27 + v14;
  if ( v15 )
  {
    v16 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v27 <= 0 )
      v16 = 0x8000000000000000LL;
  }
  v15 = __OFADD__(v13, v16);
  v17 = v13 + v16;
  if ( v15 )
  {
    v18 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v13 <= 0 )
      v18 = 0x8000000000000000LL;
  }
  else
  {
    v18 = v17;
  }
  if ( v32 != &v33 )
    _libc_free((unsigned __int64)v32);
  if ( v30 != &v31 )
    _libc_free((unsigned __int64)v30);
  return v18;
}

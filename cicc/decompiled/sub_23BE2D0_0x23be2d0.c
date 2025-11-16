// Function: sub_23BE2D0
// Address: 0x23be2d0
//
void __fastcall sub_23BE2D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, void *a5, __int64 a6, __int64 *a7)
{
  _QWORD *v8; // rax
  __int64 v9; // rax
  _QWORD v12[3]; // [rsp+20h] [rbp-120h] BYREF
  __int64 v13; // [rsp+38h] [rbp-108h] BYREF
  unsigned __int8 *v14; // [rsp+40h] [rbp-100h] BYREF
  size_t v15; // [rsp+48h] [rbp-F8h]
  __int64 v16; // [rsp+50h] [rbp-F0h]
  _BYTE v17[24]; // [rsp+58h] [rbp-E8h] BYREF
  void *v18; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v19; // [rsp+78h] [rbp-C8h]
  __int64 v20; // [rsp+80h] [rbp-C0h]
  __int64 v21; // [rsp+88h] [rbp-B8h]
  __int64 v22; // [rsp+90h] [rbp-B0h]
  __int64 v23; // [rsp+98h] [rbp-A8h]
  unsigned __int8 **v24; // [rsp+A0h] [rbp-A0h]
  _QWORD v25[2]; // [rsp+B0h] [rbp-90h] BYREF
  __int64 (__fastcall *v26)(unsigned __int64 *, const __m128i **, int); // [rsp+C0h] [rbp-80h]
  __int64 v27; // [rsp+C8h] [rbp-78h]
  char v28; // [rsp+D0h] [rbp-70h]
  _QWORD v29[2]; // [rsp+D8h] [rbp-68h] BYREF
  _QWORD v30[2]; // [rsp+E8h] [rbp-58h] BYREF
  _QWORD v31[9]; // [rsp+F8h] [rbp-48h] BYREF

  v25[0] = "*** IR Dump After {0} on {1} ***\n";
  v26 = (__int64 (__fastcall *)(unsigned __int64 *, const __m128i **, int))v31;
  v29[1] = a4;
  v29[0] = &unk_49E6618;
  v30[1] = v12;
  v30[0] = &unk_49DB108;
  v31[0] = v30;
  v12[0] = a2;
  v12[1] = a3;
  v31[1] = v29;
  v25[1] = 33;
  v27 = 2;
  v28 = 1;
  v14 = v17;
  v15 = 0;
  v16 = 20;
  v19 = 2;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0x100000000LL;
  v24 = &v14;
  v18 = &unk_49DD288;
  sub_CB5980((__int64)&v18, 0, 0, 0);
  sub_CB6840((__int64)&v18, (__int64)v25);
  v18 = &unk_49DD388;
  sub_CB5840((__int64)&v18);
  sub_CB6200(*(_QWORD *)(a1 + 40), v14, v15);
  v26 = 0;
  v18 = a5;
  v19 = a6;
  v8 = (_QWORD *)sub_22077B0(0x18u);
  if ( v8 )
  {
    *v8 = a4;
    v8[2] = a1;
    v8[1] = v12;
  }
  v25[0] = v8;
  v27 = (__int64)sub_23B7B50;
  v26 = sub_23AE880;
  sub_23B2720(&v13, a7);
  v9 = sub_23B27D0(&v13);
  sub_23BE020((__int64 *)&v18, v9 != 0, (__int64)v25);
  if ( v13 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
  if ( v26 )
    v26(v25, (const __m128i **)v25, 3);
  sub_904010(*(_QWORD *)(a1 + 40), "\n");
  if ( v14 != v17 )
    _libc_free((unsigned __int64)v14);
}

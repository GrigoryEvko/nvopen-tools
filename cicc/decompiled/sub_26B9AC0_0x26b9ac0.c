// Function: sub_26B9AC0
// Address: 0x26b9ac0
//
__int64 __fastcall sub_26B9AC0(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  char v6; // al
  char v7; // al
  __int64 v8; // rdx
  __int64 v9; // rcx
  _QWORD *v10; // r8
  char v11; // r13
  __int64 v12; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  _QWORD v22[2]; // [rsp+20h] [rbp-110h] BYREF
  __int64 (__fastcall *v23)(_QWORD *, _QWORD *, int); // [rsp+30h] [rbp-100h]
  __int64 (__fastcall *v24)(__int64 *, __int64); // [rsp+38h] [rbp-F8h]
  _QWORD v25[2]; // [rsp+40h] [rbp-F0h] BYREF
  __int64 (__fastcall *v26)(_QWORD *, _QWORD *, int); // [rsp+50h] [rbp-E0h]
  __int64 (__fastcall *v27)(__int64 *, __int64); // [rsp+58h] [rbp-D8h]
  _QWORD v28[2]; // [rsp+60h] [rbp-D0h] BYREF
  __int64 (__fastcall *v29)(_QWORD *, _QWORD *, int); // [rsp+70h] [rbp-C0h]
  __int64 (__fastcall *v30)(__int64 *, __int64); // [rsp+78h] [rbp-B8h]
  _QWORD v31[2]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 (__fastcall *v32)(_QWORD *, _QWORD *, int); // [rsp+90h] [rbp-A0h]
  __int64 (__fastcall *v33)(__int64 *, __int64); // [rsp+98h] [rbp-98h]
  __int64 v34; // [rsp+A0h] [rbp-90h] BYREF
  _BYTE *v35; // [rsp+A8h] [rbp-88h]
  __int64 v36; // [rsp+B0h] [rbp-80h]
  __int64 (__fastcall *v37)(__int64 *, __int64); // [rsp+B8h] [rbp-78h]
  _BYTE v38[16]; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v39; // [rsp+D0h] [rbp-60h] BYREF
  _BYTE *v40; // [rsp+D8h] [rbp-58h]
  __int64 v41; // [rsp+E0h] [rbp-50h]
  int v42; // [rsp+E8h] [rbp-48h]
  char v43; // [rsp+ECh] [rbp-44h]
  _BYTE v44[64]; // [rsp+F0h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  v6 = *a2;
  v36 = (__int64)sub_26B71D0;
  v37 = sub_26B71B0;
  v33 = sub_26B7130;
  v30 = sub_26B7190;
  v24 = sub_26B7150;
  v32 = sub_26B7200;
  v29 = sub_26B7230;
  v27 = sub_26B7170;
  v26 = sub_26B7260;
  v23 = sub_26B7290;
  v34 = v5;
  v31[0] = v5;
  v28[0] = v5;
  v25[0] = v5;
  v22[0] = v5;
  v7 = sub_26B77E0(
         (__int64 *)a3,
         (__int64 **)(a3 + 312),
         v5,
         (__int64)v22,
         (__int64)v25,
         (__int64)v28,
         (__int64)v31,
         (__int64)&v34,
         v6);
  v10 = v25;
  v11 = v7;
  if ( v23 )
  {
    ((void (__fastcall *)(_QWORD *, _QWORD *, __int64, __int64, _QWORD *))v23)(v22, v22, 3, v9, v25);
    v10 = v25;
  }
  if ( v26 )
    v26(v25, v25, 3);
  if ( v29 )
    ((void (__fastcall *)(_QWORD *, _QWORD *, __int64, __int64, _QWORD *))v29)(v28, v28, 3, v9, v10);
  if ( v32 )
    ((void (__fastcall *)(_QWORD *, _QWORD *, __int64, __int64, _QWORD *))v32)(v31, v31, 3, v9, v10);
  if ( v36 )
    ((void (__fastcall *)(__int64 *, __int64 *, __int64, __int64, _QWORD *))v36)(&v34, &v34, 3, v9, v10);
  v12 = a1 + 32;
  if ( v11 )
  {
    BYTE4(v37) = 1;
    v35 = v38;
    v34 = 0;
    v36 = 2;
    LODWORD(v37) = 0;
    v39 = 0;
    v40 = v44;
    v41 = 2;
    v42 = 0;
    v43 = 1;
    sub_26B73D0((__int64)&v34, (__int64)&unk_4F81450, v8, (__int64)v38, (__int64)v10, v12);
    sub_26B73D0((__int64)&v34, (__int64)&unk_4F8FBC8, v14, v15, v16, v17);
    sub_26B73D0((__int64)&v34, (__int64)&unk_4F82418, v18, v19, v20, v21);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v38, (__int64)&v34);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v44, (__int64)&v39);
    if ( !v43 )
      _libc_free((unsigned __int64)v40);
    if ( !BYTE4(v37) )
      _libc_free((unsigned __int64)v35);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v12;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}

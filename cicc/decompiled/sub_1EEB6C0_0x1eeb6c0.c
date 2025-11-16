// Function: sub_1EEB6C0
// Address: 0x1eeb6c0
//
__int64 __fastcall sub_1EEB6C0(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // rdx
  int v6; // ecx
  unsigned int v7; // r8d
  int v8; // r9d
  void (*v9)(); // rax
  unsigned __int64 v11; // [rsp+0h] [rbp-110h] BYREF
  __int64 v12; // [rsp+8h] [rbp-108h]
  int v13; // [rsp+10h] [rbp-100h]
  _BYTE v14[24]; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v15; // [rsp+38h] [rbp-D8h]
  __int64 v16; // [rsp+40h] [rbp-D0h]
  int v17; // [rsp+48h] [rbp-C8h]
  char v18; // [rsp+4Ch] [rbp-C4h]
  _BYTE *v19; // [rsp+50h] [rbp-C0h]
  __int64 v20; // [rsp+58h] [rbp-B8h]
  _BYTE v21[32]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v22; // [rsp+80h] [rbp-90h]
  unsigned __int64 v23; // [rsp+88h] [rbp-88h]
  __int64 v24; // [rsp+90h] [rbp-80h]
  int v25; // [rsp+98h] [rbp-78h]
  unsigned __int64 v26; // [rsp+A0h] [rbp-70h]
  __int64 v27; // [rsp+A8h] [rbp-68h]
  int v28; // [rsp+B0h] [rbp-60h]
  unsigned __int64 v29; // [rsp+B8h] [rbp-58h]
  __int64 v30; // [rsp+C0h] [rbp-50h]
  int v31; // [rsp+C8h] [rbp-48h]
  unsigned __int64 v32; // [rsp+D0h] [rbp-40h]
  __int64 v33; // [rsp+D8h] [rbp-38h]
  int v34; // [rsp+E0h] [rbp-30h]

  v2 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 48LL);
  if ( v2 == sub_1D90020 )
  {
    v15 = 0;
    v19 = v21;
    v18 = 0;
    v16 = 0;
    v17 = 0;
    v20 = 0x200000000LL;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v11 = 0;
    v12 = 0;
    v13 = 0;
    BUG();
  }
  v3 = v2();
  v18 = 0;
  v4 = v3;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v19 = v21;
  v20 = 0x200000000LL;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  (*(void (__fastcall **)(__int64, __int64, unsigned __int64 *, _BYTE *))(*(_QWORD *)v3 + 192LL))(v3, a2, &v11, v14);
  v9 = *(void (**)())(*(_QWORD *)v4 + 200LL);
  if ( v9 != nullsub_733 )
    ((void (__fastcall *)(__int64, __int64, _BYTE *))v9)(v4, a2, v14);
  sub_1EEB600((_QWORD *)a2, (__int64)v14, v5, v6, v7, v8);
  _libc_free(v11);
  _libc_free(v32);
  _libc_free(v29);
  _libc_free(v26);
  _libc_free(v23);
  if ( v19 != v21 )
    _libc_free((unsigned __int64)v19);
  return 1;
}

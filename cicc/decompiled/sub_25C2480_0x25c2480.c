// Function: sub_25C2480
// Address: 0x25c2480
//
void __fastcall sub_25C2480(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 (__fastcall *v4)(unsigned __int64 *, __int64 *, int); // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rbx
  unsigned __int64 v9; // r12
  void (__fastcall *v10)(unsigned __int64, unsigned __int64, __int64); // rax
  void (__fastcall *v11)(unsigned __int64, unsigned __int64, __int64); // rax
  void (__fastcall *v12)(unsigned __int64, unsigned __int64, __int64); // rax
  _BYTE v13[16]; // [rsp+0h] [rbp-320h] BYREF
  __int64 (__fastcall *v14)(_BYTE *, __int64, int); // [rsp+10h] [rbp-310h]
  __int64 (__fastcall *v15)(__int64, __int64); // [rsp+18h] [rbp-308h]
  _QWORD v16[2]; // [rsp+20h] [rbp-300h] BYREF
  __int64 (__fastcall *v17)(unsigned __int64 *, __int64 *, int); // [rsp+30h] [rbp-2F0h]
  char (__fastcall *v18)(__int64 *, unsigned __int8 *); // [rsp+38h] [rbp-2E8h]
  _BYTE v19[16]; // [rsp+40h] [rbp-2E0h] BYREF
  __int64 (__fastcall *v20)(_BYTE *, __int64, int); // [rsp+50h] [rbp-2D0h]
  unsigned __int64 (__fastcall *v21)(__int64, __int64); // [rsp+58h] [rbp-2C8h]
  _BYTE v22[16]; // [rsp+60h] [rbp-2C0h] BYREF
  void (__fastcall *v23)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-2B0h]
  __int64 (__fastcall *v24)(__int64, __int64); // [rsp+78h] [rbp-2A8h]
  _BYTE v25[16]; // [rsp+80h] [rbp-2A0h] BYREF
  __int64 (__fastcall *v26)(unsigned __int64 *, __int64 *, int); // [rsp+90h] [rbp-290h]
  char (__fastcall *v27)(__int64 *, unsigned __int8 *); // [rsp+98h] [rbp-288h]
  _BYTE v28[16]; // [rsp+A0h] [rbp-280h] BYREF
  void (__fastcall *v29)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-270h]
  unsigned __int64 (__fastcall *v30)(__int64, __int64); // [rsp+B8h] [rbp-268h]
  int v31; // [rsp+C0h] [rbp-260h]
  char v32; // [rsp+C4h] [rbp-25Ch]
  _BYTE v33[8]; // [rsp+D0h] [rbp-250h] BYREF
  __int64 v34; // [rsp+D8h] [rbp-248h]
  unsigned int v35; // [rsp+E8h] [rbp-238h]
  char *v36; // [rsp+F0h] [rbp-230h]
  char v37; // [rsp+100h] [rbp-220h] BYREF
  __m128i v38; // [rsp+140h] [rbp-1E0h] BYREF
  _BYTE v39[464]; // [rsp+150h] [rbp-1D0h] BYREF

  v38.m128i_i64[1] = 0x400000000LL;
  v15 = sub_25BCB90;
  v14 = (__int64 (__fastcall *)(_BYTE *, __int64, int))sub_25BC760;
  v38.m128i_i64[0] = (__int64)v39;
  sub_25BFBE0((__int64)v33, a1);
  v17 = 0;
  v2 = sub_22077B0(0x70u);
  v3 = v2;
  if ( v2 )
    sub_25BFBE0(v2, (__int64)v33);
  v16[0] = v3;
  v21 = sub_25BCC10;
  v20 = (__int64 (__fastcall *)(_BYTE *, __int64, int))sub_25BC770;
  v18 = sub_25BD860;
  v4 = sub_25BFD10;
  v17 = sub_25BFD10;
  v23 = 0;
  if ( v14 )
  {
    v14(v22, (__int64)v13, 2);
    v26 = 0;
    v24 = v15;
    v23 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v14;
    v4 = v17;
    if ( !v17 )
      goto LABEL_6;
  }
  else
  {
    v26 = 0;
  }
  v4((unsigned __int64 *)v25, v16, 2);
  v27 = v18;
  v26 = v17;
LABEL_6:
  v29 = 0;
  if ( v20 )
  {
    v20(v28, (__int64)v19, 2);
    v30 = v21;
    v29 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))v20;
  }
  v31 = 6;
  v32 = 0;
  sub_25BE0C0((__int64)&v38, (unsigned __int64)v22);
  if ( v29 )
    v29(v28, v28, 3);
  if ( v26 )
    v26((unsigned __int64 *)v25, (__int64 *)v25, 3);
  if ( v23 )
    v23(v22, v22, 3);
  if ( v20 )
    v20(v19, (__int64)v19, 3);
  if ( v17 )
    v17(v16, v16, 3);
  if ( v36 != &v37 )
    _libc_free((unsigned __int64)v36);
  sub_C7D6A0(v34, 8LL * v35, 8);
  if ( v14 )
    v14(v13, (__int64)v13, 3);
  sub_25C1030(&v38, a1, a2, v5, v6, v7);
  v8 = v38.m128i_i64[0];
  v9 = v38.m128i_i64[0] + 104LL * v38.m128i_u32[2];
  if ( v38.m128i_i64[0] != v9 )
  {
    do
    {
      v10 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v9 - 24);
      v9 -= 104LL;
      if ( v10 )
        v10(v9 + 64, v9 + 64, 3);
      v11 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v9 + 48);
      if ( v11 )
        v11(v9 + 32, v9 + 32, 3);
      v12 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v9 + 16);
      if ( v12 )
        v12(v9, v9, 3);
    }
    while ( v8 != v9 );
    v9 = v38.m128i_i64[0];
  }
  if ( (_BYTE *)v9 != v39 )
    _libc_free(v9);
}

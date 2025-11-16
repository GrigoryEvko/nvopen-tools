// Function: sub_A01490
// Address: 0xa01490
//
__int64 __fastcall sub_A01490(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        unsigned __int8 a4,
        unsigned __int8 a5,
        __int64 a6,
        __m128i a7)
{
  bool v10; // zf
  __int64 v11; // r9
  char v12; // dl
  void (__fastcall *v14)(_QWORD *, __int64, __int64); // rax
  void (__fastcall *v15)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v16)(_BYTE *, __int64, __int64); // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // [rsp+0h] [rbp-E0h]
  __int64 v20; // [rsp+0h] [rbp-E0h]
  __int64 v21; // [rsp+0h] [rbp-E0h]
  const __m128i *v23[2]; // [rsp+10h] [rbp-D0h] BYREF
  _QWORD v24[2]; // [rsp+30h] [rbp-B0h] BYREF
  void (__fastcall *v25)(_QWORD *, _QWORD *, __int64); // [rsp+40h] [rbp-A0h]
  __int64 v26; // [rsp+48h] [rbp-98h]
  char v27; // [rsp+50h] [rbp-90h]
  _BYTE v28[16]; // [rsp+58h] [rbp-88h] BYREF
  void (__fastcall *v29)(_BYTE *, _BYTE *, __int64); // [rsp+68h] [rbp-78h]
  __int64 v30; // [rsp+70h] [rbp-70h]
  char v31; // [rsp+78h] [rbp-68h]
  _BYTE v32[16]; // [rsp+80h] [rbp-60h] BYREF
  void (__fastcall *v33)(_BYTE *, _BYTE *, __int64); // [rsp+90h] [rbp-50h]
  __int64 v34; // [rsp+98h] [rbp-48h]
  char v35; // [rsp+A0h] [rbp-40h]

  v10 = *(_BYTE *)(a6 + 32) == 0;
  v27 = 0;
  if ( v10 )
  {
    v10 = *(_BYTE *)(a6 + 72) == 0;
    v31 = 0;
    if ( v10 )
      goto LABEL_3;
  }
  else
  {
    v25 = 0;
    v14 = *(void (__fastcall **)(_QWORD *, __int64, __int64))(a6 + 16);
    if ( v14 )
    {
      v19 = a6;
      v14(v24, a6, 2);
      a6 = v19;
      v26 = *(_QWORD *)(v19 + 24);
      v25 = *(void (__fastcall **)(_QWORD *, _QWORD *, __int64))(v19 + 16);
    }
    v10 = *(_BYTE *)(a6 + 72) == 0;
    v27 = 1;
    v31 = 0;
    if ( v10 )
    {
LABEL_3:
      v10 = *(_BYTE *)(a6 + 112) == 0;
      v35 = 0;
      if ( v10 )
        goto LABEL_4;
      goto LABEL_15;
    }
  }
  v15 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a6 + 56);
  v29 = 0;
  if ( v15 )
  {
    v20 = a6;
    v15(v28, a6 + 40, 2);
    a6 = v20;
    v30 = *(_QWORD *)(v20 + 64);
    v29 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(v20 + 56);
  }
  v10 = *(_BYTE *)(a6 + 112) == 0;
  v31 = 1;
  v35 = 0;
  if ( !v10 )
  {
LABEL_15:
    v16 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a6 + 96);
    v33 = 0;
    if ( v16 )
    {
      v21 = a6;
      v16(v32, a6 + 80, 2);
      v34 = *(_QWORD *)(v21 + 104);
      v33 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(v21 + 96);
    }
    v35 = 1;
  }
LABEL_4:
  *(double *)a7.m128i_i64 = sub_C7EC60(v23, *a2);
  sub_A011E0(a1, a3, a4, a5, (__int64)v24, v11, a7, v23[0], (unsigned __int64)v23[1]);
  if ( v35 )
  {
    v35 = 0;
    if ( v33 )
      v33(v32, v32, 3);
  }
  if ( v31 )
  {
    v31 = 0;
    if ( v29 )
      v29(v28, v28, 3);
  }
  if ( v27 )
  {
    v27 = 0;
    if ( v25 )
      v25(v24, v24, 3);
  }
  v12 = *(_BYTE *)(a1 + 8) & 1;
  *(_BYTE *)(a1 + 8) = (2 * v12) | *(_BYTE *)(a1 + 8) & 0xFD;
  if ( !v12 )
  {
    v17 = *(_QWORD *)a1;
    v18 = *a2;
    *a2 = 0;
    v24[0] = v18;
    sub_BAA730(v17, v24);
    if ( v24[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v24[0] + 8LL))(v24[0]);
  }
  return a1;
}

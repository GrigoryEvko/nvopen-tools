// Function: sub_A01950
// Address: 0xa01950
//
__int64 __fastcall sub_A01950(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        const __m128i *a8,
        unsigned __int64 a9)
{
  char v10; // dl
  char v11; // al
  __int64 v12; // rax
  bool v14; // zf
  char v15; // al
  void (__fastcall *v16)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v17)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v18)(_BYTE *, __int64, __int64); // rax
  __m128i v19[4]; // [rsp+0h] [rbp-100h] BYREF
  char v20; // [rsp+40h] [rbp-C0h]
  _BYTE v21[16]; // [rsp+50h] [rbp-B0h] BYREF
  void (__fastcall *v22)(_BYTE *, _BYTE *, __int64); // [rsp+60h] [rbp-A0h]
  __int64 v23; // [rsp+68h] [rbp-98h]
  char v24; // [rsp+70h] [rbp-90h]
  _BYTE v25[16]; // [rsp+78h] [rbp-88h] BYREF
  void (__fastcall *v26)(_BYTE *, _BYTE *, __int64); // [rsp+88h] [rbp-78h]
  __int64 v27; // [rsp+90h] [rbp-70h]
  char v28; // [rsp+98h] [rbp-68h]
  _BYTE v29[16]; // [rsp+A0h] [rbp-60h] BYREF
  void (__fastcall *v30)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-50h]
  __int64 v31; // [rsp+B8h] [rbp-48h]
  char v32; // [rsp+C0h] [rbp-40h]

  sub_9D5100(v19, a2, a3, a4, a5, a6, a7, a8, a9);
  v10 = v20 & 1;
  v11 = (2 * (v20 & 1)) | v20 & 0xFD;
  v20 = v11;
  if ( !v10 )
  {
    v14 = *(_BYTE *)(a3 + 32) == 0;
    v24 = 0;
    if ( v14 )
    {
      v14 = *(_BYTE *)(a3 + 72) == 0;
      v28 = 0;
      if ( v14 )
        goto LABEL_8;
    }
    else
    {
      v22 = 0;
      v16 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a3 + 16);
      if ( v16 )
      {
        v16(v21, a3, 2);
        v23 = *(_QWORD *)(a3 + 24);
        v22 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a3 + 16);
      }
      v14 = *(_BYTE *)(a3 + 72) == 0;
      v24 = 1;
      v28 = 0;
      if ( v14 )
      {
LABEL_8:
        v14 = *(_BYTE *)(a3 + 112) == 0;
        v32 = 0;
        if ( v14 )
          goto LABEL_9;
        goto LABEL_21;
      }
    }
    v17 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a3 + 56);
    v26 = 0;
    if ( v17 )
    {
      v17(v25, a3 + 40, 2);
      v27 = *(_QWORD *)(a3 + 64);
      v26 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a3 + 56);
    }
    v14 = *(_BYTE *)(a3 + 112) == 0;
    v28 = 1;
    v32 = 0;
    if ( v14 )
    {
LABEL_9:
      sub_A01750(a1, v19, a2, (__int64)v21);
      if ( v32 && (v32 = 0, v30) )
      {
        v30(v29, v29, 3);
        if ( !v28 )
          goto LABEL_11;
      }
      else if ( !v28 )
      {
        goto LABEL_11;
      }
      v28 = 0;
      if ( v26 )
      {
        v26(v25, v25, 3);
        if ( !v24 )
          goto LABEL_12;
        goto LABEL_28;
      }
LABEL_11:
      if ( !v24 )
        goto LABEL_12;
LABEL_28:
      v24 = 0;
      if ( v22 )
      {
        v22(v21, v21, 3);
        v15 = v20;
        if ( (v20 & 2) != 0 )
LABEL_30:
          sub_9D52C0(v19);
LABEL_13:
        if ( (v15 & 1) == 0 )
          return a1;
        goto LABEL_3;
      }
LABEL_12:
      v15 = v20;
      if ( (v20 & 2) != 0 )
        goto LABEL_30;
      goto LABEL_13;
    }
LABEL_21:
    v18 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a3 + 96);
    v30 = 0;
    if ( v18 )
    {
      v18(v29, a3 + 80, 2);
      v31 = *(_QWORD *)(a3 + 104);
      v30 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a3 + 96);
    }
    v32 = 1;
    goto LABEL_9;
  }
  *(_BYTE *)(a1 + 8) |= 3u;
  v20 = v11 & 0xFD;
  v12 = v19[0].m128i_i64[0];
  v19[0].m128i_i64[0] = 0;
  *(_QWORD *)a1 = v12 & 0xFFFFFFFFFFFFFFFELL;
LABEL_3:
  if ( v19[0].m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v19[0].m128i_i64[0] + 8LL))(v19[0].m128i_i64[0]);
  return a1;
}

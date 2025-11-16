// Function: sub_A011E0
// Address: 0xa011e0
//
__int64 __fastcall sub_A011E0(
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
  unsigned __int8 v19; // [rsp+8h] [rbp-108h]
  unsigned __int8 v20; // [rsp+Ch] [rbp-104h]
  __m128i v21[4]; // [rsp+10h] [rbp-100h] BYREF
  char v22; // [rsp+50h] [rbp-C0h]
  _BYTE v23[16]; // [rsp+60h] [rbp-B0h] BYREF
  void (__fastcall *v24)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-A0h]
  __int64 v25; // [rsp+78h] [rbp-98h]
  char v26; // [rsp+80h] [rbp-90h]
  _BYTE v27[16]; // [rsp+88h] [rbp-88h] BYREF
  void (__fastcall *v28)(_BYTE *, _BYTE *, __int64); // [rsp+98h] [rbp-78h]
  __int64 v29; // [rsp+A0h] [rbp-70h]
  char v30; // [rsp+A8h] [rbp-68h]
  _BYTE v31[16]; // [rsp+B0h] [rbp-60h] BYREF
  void (__fastcall *v32)(_BYTE *, _BYTE *, __int64); // [rsp+C0h] [rbp-50h]
  __int64 v33; // [rsp+C8h] [rbp-48h]
  char v34; // [rsp+D0h] [rbp-40h]

  v20 = a3;
  v19 = a4;
  sub_9D5100(v21, a2, a3, a4, a5, a6, a7, a8, a9);
  v10 = v22 & 1;
  v11 = (2 * (v22 & 1)) | v22 & 0xFD;
  v22 = v11;
  if ( !v10 )
  {
    v14 = *(_BYTE *)(a5 + 32) == 0;
    v26 = 0;
    if ( v14 )
    {
      v14 = *(_BYTE *)(a5 + 72) == 0;
      v30 = 0;
      if ( v14 )
        goto LABEL_8;
    }
    else
    {
      v24 = 0;
      v16 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a5 + 16);
      if ( v16 )
      {
        v16(v23, a5, 2);
        v25 = *(_QWORD *)(a5 + 24);
        v24 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a5 + 16);
      }
      v14 = *(_BYTE *)(a5 + 72) == 0;
      v26 = 1;
      v30 = 0;
      if ( v14 )
      {
LABEL_8:
        v14 = *(_BYTE *)(a5 + 112) == 0;
        v34 = 0;
        if ( v14 )
          goto LABEL_9;
        goto LABEL_21;
      }
    }
    v17 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a5 + 56);
    v28 = 0;
    if ( v17 )
    {
      v17(v27, a5 + 40, 2);
      v29 = *(_QWORD *)(a5 + 64);
      v28 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a5 + 56);
    }
    v14 = *(_BYTE *)(a5 + 112) == 0;
    v30 = 1;
    v34 = 0;
    if ( v14 )
    {
LABEL_9:
      sub_A00F90(a1, v21, a2, v20, v19, (__int64)v23);
      if ( v34 && (v34 = 0, v32) )
      {
        v32(v31, v31, 3);
        if ( !v30 )
          goto LABEL_11;
      }
      else if ( !v30 )
      {
        goto LABEL_11;
      }
      v30 = 0;
      if ( v28 )
      {
        v28(v27, v27, 3);
        if ( !v26 )
          goto LABEL_12;
        goto LABEL_28;
      }
LABEL_11:
      if ( !v26 )
        goto LABEL_12;
LABEL_28:
      v26 = 0;
      if ( v24 )
      {
        v24(v23, v23, 3);
        v15 = v22;
        if ( (v22 & 2) != 0 )
LABEL_30:
          sub_9D52C0(v21);
LABEL_13:
        if ( (v15 & 1) == 0 )
          return a1;
        goto LABEL_3;
      }
LABEL_12:
      v15 = v22;
      if ( (v22 & 2) != 0 )
        goto LABEL_30;
      goto LABEL_13;
    }
LABEL_21:
    v18 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a5 + 96);
    v32 = 0;
    if ( v18 )
    {
      v18(v31, a5 + 80, 2);
      v33 = *(_QWORD *)(a5 + 104);
      v32 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a5 + 96);
    }
    v34 = 1;
    goto LABEL_9;
  }
  *(_BYTE *)(a1 + 8) |= 3u;
  v22 = v11 & 0xFD;
  v12 = v21[0].m128i_i64[0];
  v21[0].m128i_i64[0] = 0;
  *(_QWORD *)a1 = v12 & 0xFFFFFFFFFFFFFFFELL;
LABEL_3:
  if ( v21[0].m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v21[0].m128i_i64[0] + 8LL))(v21[0].m128i_i64[0]);
  return a1;
}

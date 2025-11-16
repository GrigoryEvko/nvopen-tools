// Function: sub_A00F90
// Address: 0xa00f90
//
__int64 __fastcall sub_A00F90(__int64 a1, _QWORD *a2, __int64 a3, unsigned __int8 a4, unsigned __int8 a5, __int64 a6)
{
  bool v8; // zf
  void (__fastcall *v10)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v11)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v12)(_BYTE *, __int64, __int64); // rax
  unsigned __int8 v13; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v14; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v15; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v16; // [rsp+Ch] [rbp-B4h]
  unsigned __int8 v17; // [rsp+Ch] [rbp-B4h]
  unsigned __int8 v18; // [rsp+Ch] [rbp-B4h]
  _BYTE v19[16]; // [rsp+10h] [rbp-B0h] BYREF
  void (__fastcall *v20)(_BYTE *, _BYTE *, __int64); // [rsp+20h] [rbp-A0h]
  __int64 v21; // [rsp+28h] [rbp-98h]
  char v22; // [rsp+30h] [rbp-90h]
  _BYTE v23[16]; // [rsp+38h] [rbp-88h] BYREF
  void (__fastcall *v24)(_BYTE *, _BYTE *, __int64); // [rsp+48h] [rbp-78h]
  __int64 v25; // [rsp+50h] [rbp-70h]
  char v26; // [rsp+58h] [rbp-68h]
  _BYTE v27[16]; // [rsp+60h] [rbp-60h] BYREF
  void (__fastcall *v28)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-50h]
  __int64 v29; // [rsp+78h] [rbp-48h]
  char v30; // [rsp+80h] [rbp-40h]

  v8 = *(_BYTE *)(a6 + 32) == 0;
  v22 = 0;
  if ( v8 )
  {
    v8 = *(_BYTE *)(a6 + 72) == 0;
    v26 = 0;
    if ( v8 )
      goto LABEL_3;
  }
  else
  {
    v20 = 0;
    v10 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a6 + 16);
    if ( v10 )
    {
      v13 = a5;
      v16 = a4;
      v10(v19, a6, 2);
      a5 = v13;
      a4 = v16;
      v21 = *(_QWORD *)(a6 + 24);
      v20 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a6 + 16);
    }
    v8 = *(_BYTE *)(a6 + 72) == 0;
    v22 = 1;
    v26 = 0;
    if ( v8 )
    {
LABEL_3:
      v8 = *(_BYTE *)(a6 + 112) == 0;
      v30 = 0;
      if ( v8 )
        goto LABEL_4;
      goto LABEL_14;
    }
  }
  v11 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a6 + 56);
  v24 = 0;
  if ( v11 )
  {
    v14 = a5;
    v17 = a4;
    v11(v23, a6 + 40, 2);
    a5 = v14;
    a4 = v17;
    v25 = *(_QWORD *)(a6 + 64);
    v24 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a6 + 56);
  }
  v8 = *(_BYTE *)(a6 + 112) == 0;
  v26 = 1;
  v30 = 0;
  if ( !v8 )
  {
LABEL_14:
    v12 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a6 + 96);
    v28 = 0;
    if ( v12 )
    {
      v15 = a5;
      v18 = a4;
      v12(v27, a6 + 80, 2);
      a5 = v15;
      a4 = v18;
      v29 = *(_QWORD *)(a6 + 104);
      v28 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a6 + 96);
    }
    v30 = 1;
  }
LABEL_4:
  sub_9FF220(a1, a2, a3, 0, a4, a5, (__int64)v19);
  if ( v30 )
  {
    v30 = 0;
    if ( v28 )
      v28(v27, v27, 3);
  }
  if ( v26 )
  {
    v26 = 0;
    if ( v24 )
      v24(v23, v23, 3);
  }
  if ( v22 )
  {
    v22 = 0;
    if ( v20 )
      v20(v19, v19, 3);
  }
  return a1;
}

// Function: sub_A01750
// Address: 0xa01750
//
__int64 __fastcall sub_A01750(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  bool v6; // zf
  void (__fastcall *v8)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v9)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v10)(_BYTE *, __int64, __int64); // rax
  _BYTE v11[16]; // [rsp+0h] [rbp-B0h] BYREF
  void (__fastcall *v12)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-A0h]
  __int64 v13; // [rsp+18h] [rbp-98h]
  char v14; // [rsp+20h] [rbp-90h]
  _BYTE v15[16]; // [rsp+28h] [rbp-88h] BYREF
  void (__fastcall *v16)(_BYTE *, _BYTE *, __int64); // [rsp+38h] [rbp-78h]
  __int64 v17; // [rsp+40h] [rbp-70h]
  char v18; // [rsp+48h] [rbp-68h]
  _BYTE v19[16]; // [rsp+50h] [rbp-60h] BYREF
  void (__fastcall *v20)(_BYTE *, _BYTE *, __int64); // [rsp+60h] [rbp-50h]
  __int64 v21; // [rsp+68h] [rbp-48h]
  char v22; // [rsp+70h] [rbp-40h]

  v6 = *(_BYTE *)(a4 + 32) == 0;
  v14 = 0;
  if ( v6 )
  {
    v6 = *(_BYTE *)(a4 + 72) == 0;
    v18 = 0;
    if ( v6 )
      goto LABEL_3;
  }
  else
  {
    v12 = 0;
    v8 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a4 + 16);
    if ( v8 )
    {
      v8(v11, a4, 2);
      v13 = *(_QWORD *)(a4 + 24);
      v12 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a4 + 16);
    }
    v6 = *(_BYTE *)(a4 + 72) == 0;
    v14 = 1;
    v18 = 0;
    if ( v6 )
    {
LABEL_3:
      v6 = *(_BYTE *)(a4 + 112) == 0;
      v22 = 0;
      if ( v6 )
        goto LABEL_4;
      goto LABEL_14;
    }
  }
  v9 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a4 + 56);
  v16 = 0;
  if ( v9 )
  {
    v9(v15, a4 + 40, 2);
    v17 = *(_QWORD *)(a4 + 64);
    v16 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a4 + 56);
  }
  v6 = *(_BYTE *)(a4 + 112) == 0;
  v18 = 1;
  v22 = 0;
  if ( !v6 )
  {
LABEL_14:
    v10 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a4 + 96);
    v20 = 0;
    if ( v10 )
    {
      v10(v19, a4 + 80, 2);
      v21 = *(_QWORD *)(a4 + 104);
      v20 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a4 + 96);
    }
    v22 = 1;
  }
LABEL_4:
  sub_9FF220(a1, a2, a3, 1, 0, 0, (__int64)v11);
  if ( v22 )
  {
    v22 = 0;
    if ( v20 )
      v20(v19, v19, 3);
  }
  if ( v18 )
  {
    v18 = 0;
    if ( v16 )
      v16(v15, v15, 3);
  }
  if ( v14 )
  {
    v14 = 0;
    if ( v12 )
      v12(v11, v11, 3);
  }
  return a1;
}

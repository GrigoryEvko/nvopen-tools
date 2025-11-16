// Function: sub_25BAEF0
// Address: 0x25baef0
//
_QWORD *__fastcall sub_25BAEF0(_QWORD *a1, int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r15
  _QWORD *v7; // r14
  int v8; // eax
  __int64 (__fastcall **v10)(); // [rsp+0h] [rbp-F0h] BYREF
  __int64 v11; // [rsp+8h] [rbp-E8h]
  void *v12; // [rsp+10h] [rbp-E0h]
  int v13; // [rsp+18h] [rbp-D8h]
  __int64 v14; // [rsp+20h] [rbp-D0h]
  __int64 v15; // [rsp+28h] [rbp-C8h]
  __int64 v16; // [rsp+30h] [rbp-C0h]
  __int64 *v17; // [rsp+38h] [rbp-B8h]
  __int64 v18; // [rsp+40h] [rbp-B0h]
  __int64 v19; // [rsp+48h] [rbp-A8h]
  __int64 v20; // [rsp+50h] [rbp-A0h]
  int v21; // [rsp+58h] [rbp-98h]
  __int64 v22; // [rsp+60h] [rbp-90h]
  __int64 v23; // [rsp+68h] [rbp-88h] BYREF
  __int64 *v24; // [rsp+70h] [rbp-80h]
  __int64 v25; // [rsp+78h] [rbp-78h]
  __int64 v26; // [rsp+80h] [rbp-70h]
  __int64 v27; // [rsp+88h] [rbp-68h]
  int v28; // [rsp+90h] [rbp-60h]
  __int64 v29; // [rsp+98h] [rbp-58h]
  __int64 v30; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v31; // [rsp+A8h] [rbp-48h]
  __int64 v32; // [rsp+B0h] [rbp-40h]

  v6 = a1 + 4;
  v7 = a1 + 10;
  v12 = &unk_4FEFDC8;
  v17 = &v23;
  v24 = &v30;
  v8 = qword_4FEFE68;
  v11 = 0;
  if ( !(_DWORD)qword_4FEFE68 )
    v8 = *a2;
  v13 = 4;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v18 = 1;
  v19 = 0;
  v20 = 0;
  v22 = 0;
  v23 = 0;
  v25 = 1;
  v26 = 0;
  v27 = 0;
  v29 = 0;
  v30 = 0;
  LOBYTE(v31) = 0;
  v10 = off_4A1F200;
  v21 = 1065353216;
  v28 = 1065353216;
  HIDWORD(v31) = v8;
  v32 = 0;
  if ( v8 != 1 && (a2 = (int *)a3, (unsigned __int8)sub_25B93C0((__int64)&v10, a3)) )
  {
    memset(a1, 0, 0x60u);
    a4 = 0;
    a1[1] = v6;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v7;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v6;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v7;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  v10 = off_4A1F200;
  if ( v32 )
    (*(void (__fastcall **)(__int64, int *, __int64, __int64, __int64, __int64, __int64 (__fastcall **)(), __int64, void *, int, __int64, __int64, __int64, __int64 *, __int64, __int64, __int64, int, __int64, __int64, __int64 *, __int64, __int64, __int64, int, __int64, __int64, __int64))(*(_QWORD *)v32 + 56LL))(
      v32,
      a2,
      a3,
      a4,
      a5,
      a6,
      v10,
      v11,
      v12,
      v13,
      v14,
      v15,
      v16,
      v17,
      v18,
      v19,
      v20,
      v21,
      v22,
      v23,
      v24,
      v25,
      v26,
      v27,
      v28,
      v29,
      v30,
      v31);
  sub_BB9260((__int64)&v10);
  return a1;
}

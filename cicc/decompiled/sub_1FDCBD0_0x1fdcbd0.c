// Function: sub_1FDCBD0
// Address: 0x1fdcbd0
//
__int64 __fastcall sub_1FDCBD0(__int64 **a1, __int64 *a2)
{
  __int64 v4; // rax
  unsigned int v5; // eax
  unsigned int v7; // ebx
  _BOOL4 v8; // r14d
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 *v11; // rdx
  char v12; // di
  __int64 (*v13)(); // r10
  unsigned int v14; // esi
  _QWORD *v15; // rdi
  __int64 v16; // rax
  unsigned int v17; // r10d
  __int64 *v18; // rdx
  __int64 (*v19)(); // rax
  unsigned int v20; // r14d
  char v21; // al
  unsigned int v22; // r10d
  unsigned int v23; // eax
  __int64 (*v24)(); // r11
  unsigned int v25; // edx
  unsigned __int8 v26; // [rsp+Ch] [rbp-44h]
  unsigned __int8 v27; // [rsp+Ch] [rbp-44h]
  unsigned int v28; // [rsp+10h] [rbp-40h] BYREF
  __int64 v29; // [rsp+18h] [rbp-38h]

  v4 = sub_15FB7B0((__int64)a2);
  v5 = sub_1FD8F60(a1, v4);
  if ( !v5 )
    return 0;
  v7 = v5;
  v8 = sub_1FD4DC0((__int64)a1, (__int64)a2);
  LOBYTE(v9) = sub_1FD35E0((__int64)a1[12], *a2);
  v29 = v10;
  v11 = *a1;
  v12 = v9;
  v28 = v9;
  v13 = (__int64 (*)())v11[8];
  if ( v13 != sub_1FD34C0 )
  {
    v25 = ((__int64 (__fastcall *)(__int64 **, _QWORD, _QWORD, __int64, _QWORD, _BOOL4))v13)(a1, v9, v9, 162, v7, v8);
    if ( v25 )
      goto LABEL_21;
    v12 = v28;
  }
  if ( v12 )
    v14 = sub_1FD3510(v12);
  else
    v14 = sub_1F58D40((__int64)&v28);
  if ( v14 > 0x40 )
    return 0;
  v15 = (_QWORD *)sub_16498A0((__int64)a2);
  if ( v14 == 16 )
  {
    v16 = 4;
    v17 = 4;
    goto LABEL_12;
  }
  if ( v14 > 0x10 )
  {
    if ( v14 == 32 )
    {
      v16 = 5;
      v17 = 5;
      goto LABEL_12;
    }
    if ( v14 == 64 )
    {
      v16 = 6;
      v17 = 6;
      goto LABEL_12;
    }
LABEL_28:
    LODWORD(v16) = sub_1F58CC0(v15, v14);
    v17 = v16;
    if ( !(_BYTE)v16 )
      return 0;
    v18 = a1[14];
    v16 = (unsigned __int8)v16;
    goto LABEL_13;
  }
  if ( v14 == 1 )
  {
    v16 = 2;
    v17 = 2;
    goto LABEL_12;
  }
  if ( v14 != 8 )
    goto LABEL_28;
  v16 = 3;
  v17 = 3;
LABEL_12:
  v18 = a1[14];
LABEL_13:
  if ( !v18[v16 + 15] )
    return 0;
  v19 = (__int64 (*)())(*a1)[8];
  if ( v19 == sub_1FD34C0 )
    return 0;
  v26 = v17;
  v20 = ((__int64 (__fastcall *)(__int64 **, _QWORD, _QWORD, __int64, _QWORD, _BOOL4))v19)(
          a1,
          (unsigned __int8)v28,
          v17,
          158,
          v7,
          v8);
  if ( !v20 )
    return 0;
  if ( (_BYTE)v28 )
  {
    v21 = sub_1FD3510(v28);
  }
  else
  {
    v21 = sub_1F58D40((__int64)&v28);
    v22 = v26;
  }
  v27 = v22;
  v23 = sub_1FDC040(a1, v22, 0x78u, v20, 1u, 1LL << (v21 - 1), v22);
  if ( !v23 )
    return 0;
  v24 = (__int64 (*)())(*a1)[8];
  if ( v24 == sub_1FD34C0 )
    return 0;
  v25 = ((__int64 (__fastcall *)(__int64 **, _QWORD, _QWORD, __int64, _QWORD, __int64))v24)(
          a1,
          v27,
          (unsigned __int8)v28,
          158,
          v23,
          1);
  if ( !v25 )
    return 0;
LABEL_21:
  sub_1FD5CC0((__int64)a1, (__int64)a2, v25, 1);
  return 1;
}

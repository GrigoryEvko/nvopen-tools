// Function: sub_2CBEA80
// Address: 0x2cbea80
//
_QWORD *__fastcall sub_2CBEA80(_QWORD *a1, __int64 a2, __int64 a3)
{
  char v3; // bl
  _QWORD *v4; // rsi
  _QWORD *v5; // rdx
  __int64 v7; // [rsp+0h] [rbp-E0h] BYREF
  int v8; // [rsp+10h] [rbp-D0h] BYREF
  unsigned __int64 v9; // [rsp+18h] [rbp-C8h]
  int *v10; // [rsp+20h] [rbp-C0h]
  int *v11; // [rsp+28h] [rbp-B8h]
  __int64 v12; // [rsp+30h] [rbp-B0h]
  int v13; // [rsp+40h] [rbp-A0h] BYREF
  unsigned __int64 v14; // [rsp+48h] [rbp-98h]
  int *v15; // [rsp+50h] [rbp-90h]
  int *v16; // [rsp+58h] [rbp-88h]
  __int64 v17; // [rsp+60h] [rbp-80h]
  int v18; // [rsp+70h] [rbp-70h] BYREF
  unsigned __int64 v19; // [rsp+78h] [rbp-68h]
  int *v20; // [rsp+80h] [rbp-60h]
  int *v21; // [rsp+88h] [rbp-58h]
  __int64 v22; // [rsp+90h] [rbp-50h]
  int v23; // [rsp+A0h] [rbp-40h] BYREF
  _QWORD *v24; // [rsp+A8h] [rbp-38h]
  int *v25; // [rsp+B0h] [rbp-30h]
  int *v26; // [rsp+B8h] [rbp-28h]
  __int64 v27; // [rsp+C0h] [rbp-20h]
  int v28; // [rsp+C8h] [rbp-18h]

  v10 = &v8;
  v11 = &v8;
  v15 = &v13;
  v16 = &v13;
  v20 = &v18;
  v21 = &v18;
  v8 = 0;
  v9 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = &v23;
  v26 = &v23;
  v27 = 0;
  v28 = 0;
  v3 = sub_2CBBE90(&v7, a3);
  sub_2CBB410(v24);
  sub_2CBA920(v19);
  sub_2CBAAF0(v14);
  sub_2CBACC0(v9);
  v4 = a1 + 4;
  v5 = a1 + 10;
  if ( v3 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v4;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v5;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v4;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[4] = &qword_4F82400;
    a1[7] = v5;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
  }
  return a1;
}

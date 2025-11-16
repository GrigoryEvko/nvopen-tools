// Function: sub_3011150
// Address: 0x3011150
//
_QWORD *__fastcall sub_3011150(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // r14
  _QWORD **v6; // rbx
  __int64 v7; // rax
  _QWORD *v8; // rdi
  _QWORD *v9; // rax
  char v10; // al
  _QWORD *v11; // rsi
  _QWORD *v12; // rdx
  _QWORD *v14; // [rsp+0h] [rbp-90h] BYREF
  __int64 v15; // [rsp+8h] [rbp-88h]
  _QWORD **v16; // [rsp+10h] [rbp-80h]
  __int64 v17; // [rsp+18h] [rbp-78h]
  __int64 v18; // [rsp+20h] [rbp-70h]
  __int64 v19; // [rsp+28h] [rbp-68h]
  __int64 v20; // [rsp+30h] [rbp-60h]
  __int64 v21; // [rsp+38h] [rbp-58h]
  __int64 v22; // [rsp+40h] [rbp-50h]
  __int64 v23; // [rsp+48h] [rbp-48h]
  __int64 v24; // [rsp+50h] [rbp-40h]
  __int64 v25; // [rsp+58h] [rbp-38h]
  __int64 v26; // [rsp+60h] [rbp-30h]

  v5 = (__int64 *)sub_B2BE50(a3);
  v6 = (_QWORD **)sub_BCB2D0(v5);
  v7 = sub_BCE3C0(v5, 0);
  v8 = *v6;
  v14 = v6;
  v16 = v6;
  v15 = v7;
  v9 = sub_BD0B90(v8, &v14, 3, 0);
  v15 = 0;
  v14 = v9;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  LOBYTE(v6) = sub_300F550((__int64)&v14, a3);
  v10 = sub_3010A20((__int64 *)&v14, a3);
  v11 = a1 + 4;
  v12 = a1 + 10;
  if ( (_BYTE)v6 || v10 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v11;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v12;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
  else
  {
    a1[1] = v11;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[4] = &qword_4F82400;
    a1[7] = v12;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    return a1;
  }
}

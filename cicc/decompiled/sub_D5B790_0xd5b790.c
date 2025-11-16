// Function: sub_D5B790
// Address: 0xd5b790
//
__int64 *__fastcall sub_D5B790(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rbx
  __int64 v4; // r15
  __int64 v5; // r9
  __int64 v6; // r8
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rdi
  _QWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v14; // [rsp+8h] [rbp-B8h] BYREF
  __int64 v15[3]; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD *v16; // [rsp+28h] [rbp-98h]
  __int64 v17[4]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v18; // [rsp+50h] [rbp-70h] BYREF
  __int64 v19; // [rsp+58h] [rbp-68h]
  __int64 v20; // [rsp+60h] [rbp-60h]
  _QWORD *v21; // [rsp+68h] [rbp-58h]
  __int64 v22; // [rsp+70h] [rbp-50h] BYREF
  __int64 v23; // [rsp+78h] [rbp-48h]
  __int64 v24; // [rsp+80h] [rbp-40h]
  _QWORD *v25; // [rsp+88h] [rbp-38h]

  v3 = (_QWORD *)a1[9];
  v4 = a1[6];
  v14 = a2;
  v5 = a1[7];
  v6 = a1[8];
  v7 = a1[3];
  v8 = a1[4];
  v22 = v4;
  v9 = a1[2];
  v10 = (_QWORD *)a1[5];
  v23 = v5;
  v19 = v7;
  v18 = v9;
  v20 = v8;
  v24 = v6;
  v25 = v3;
  v21 = v10;
  sub_D58930(v15, (__int64)&v18, &v22, &v14);
  v18 = v4;
  v11 = *v3;
  v21 = v3;
  v19 = v11;
  v20 = v11 + 512;
  v22 = v15[0];
  v12 = *v16;
  v25 = v16;
  v23 = v12;
  v24 = v12 + 512;
  return sub_D5B1D0(v17, a1, &v22, &v18);
}

// Function: sub_1CBEC10
// Address: 0x1cbec10
//
_QWORD *__fastcall sub_1CBEC10(_QWORD *a1, __int64 *a2, __int64 *a3, _QWORD *a4)
{
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // rdi
  __int64 v9; // r11
  __int64 v10; // rcx
  __int64 v11; // r10
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rax
  _QWORD v17[4]; // [rsp+0h] [rbp-B0h] BYREF
  _QWORD v18[4]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v19[4]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v20[10]; // [rsp+60h] [rbp-50h] BYREF

  v5 = a4[1];
  v6 = a4[3];
  v7 = a3[3];
  v8 = *a2;
  v9 = a4[2];
  v20[0] = *a4;
  v10 = a2[1];
  v11 = *a3;
  v20[1] = v5;
  v12 = a3[1];
  v13 = a3[2];
  v20[3] = v6;
  v14 = a2[2];
  v15 = a2[3];
  v19[3] = v7;
  v18[0] = v8;
  v18[1] = v10;
  v18[2] = v14;
  v18[3] = v15;
  v20[2] = v9;
  v19[0] = v11;
  v19[1] = v12;
  v19[2] = v13;
  sub_1CBEAB0(v17, (__int64)v18, v19, (__int64)v20);
  *a1 = v17[0];
  a1[1] = v17[1];
  a1[2] = v17[2];
  a1[3] = v17[3];
  return a1;
}

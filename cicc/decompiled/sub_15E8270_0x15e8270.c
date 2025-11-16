// Function: sub_15E8270
// Address: 0x15e8270
//
_QWORD *__fastcall sub_15E8270(__int64 *a1, __int64 *a2, __int64 *a3, unsigned int a4, __int64 a5)
{
  __int64 v7; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  _QWORD **v16; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-90h]
  __int64 v19; // [rsp+8h] [rbp-88h]
  __int64 v20[2]; // [rsp+10h] [rbp-80h] BYREF
  char v21[16]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v22; // [rsp+30h] [rbp-60h]
  _QWORD v23[10]; // [rsp+40h] [rbp-50h] BYREF

  v7 = a5;
  v9 = *a3;
  v10 = *a2;
  if ( !a5 )
  {
    v14 = *(_QWORD *)(v9 + 32);
    v18 = *a2;
    v19 = v9;
    v15 = sub_1643320(a1[3]);
    v16 = (_QWORD **)sub_16463B0(v15, (unsigned int)v14);
    v17 = sub_15A04A0(v16);
    v10 = v18;
    v9 = v19;
    v7 = v17;
  }
  v11 = a1[3];
  v20[0] = v10;
  v20[1] = v9;
  v23[0] = a2;
  v23[1] = a3;
  v12 = sub_1643350(v11);
  v23[2] = sub_159C470(v12, a4, 0);
  v22 = 257;
  v23[3] = v7;
  return sub_15E7FB0(a1, 130, (int)v23, 4, v20, 2, (__int64)v21);
}

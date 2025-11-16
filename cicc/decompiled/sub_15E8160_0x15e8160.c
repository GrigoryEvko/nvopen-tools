// Function: sub_15E8160
// Address: 0x15e8160
//
_QWORD *__fastcall sub_15E8160(__int64 *a1, __int64 *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 **v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v15; // rax
  _QWORD **v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 **v19; // [rsp+0h] [rbp-80h]
  __int64 v20; // [rsp+8h] [rbp-78h]
  __int64 **v21; // [rsp+8h] [rbp-78h]
  __int64 v24[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v25[10]; // [rsp+30h] [rbp-50h] BYREF

  v9 = *a2;
  v20 = *(_QWORD *)(*a2 + 32);
  v10 = sub_16463B0(*(_QWORD *)(*(_QWORD *)(*a2 + 24) + 24LL), (unsigned int)v20);
  v11 = (__int64 **)v10;
  if ( !a4 )
  {
    v19 = (__int64 **)v10;
    v15 = sub_1643320(a1[3]);
    v16 = (_QWORD **)sub_16463B0(v15, (unsigned int)v20);
    v17 = sub_15A04A0(v16);
    v11 = v19;
    a4 = v17;
    if ( a5 )
      goto LABEL_3;
LABEL_5:
    v21 = v11;
    v18 = sub_1599EF0(v11);
    v11 = v21;
    a5 = v18;
    goto LABEL_3;
  }
  if ( !a5 )
    goto LABEL_5;
LABEL_3:
  v12 = a1[3];
  v24[0] = (__int64)v11;
  v24[1] = v9;
  v25[0] = a2;
  v13 = sub_1643350(v12);
  v25[2] = a4;
  v25[3] = a5;
  v25[1] = sub_159C470(v13, a3, 0);
  return sub_15E7FB0(a1, 128, (int)v25, 4, v24, 2, a6);
}

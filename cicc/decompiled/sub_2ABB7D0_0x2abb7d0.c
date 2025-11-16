// Function: sub_2ABB7D0
// Address: 0x2abb7d0
//
__int64 __fastcall sub_2ABB7D0(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // [rsp+8h] [rbp-88h]
  __int64 v17; // [rsp+18h] [rbp-78h]
  char v18; // [rsp+27h] [rbp-69h]
  __int64 v19; // [rsp+28h] [rbp-68h] BYREF
  __int64 v20; // [rsp+30h] [rbp-60h] BYREF
  __int64 v21; // [rsp+38h] [rbp-58h] BYREF
  _QWORD v22[10]; // [rsp+40h] [rbp-50h] BYREF

  v8 = a1[4];
  v19 = a2;
  v9 = sub_31A6940(v8);
  if ( v9 )
    return sub_2ABB2E0(v19, v19, *a3, v9, *a1, *(_QWORD *)(a1[6] + 112));
  v10 = 0;
  v12 = sub_31A6A30(a1[4], v19);
  if ( v12 )
  {
    v13 = sub_2C47690(*a1, *(_QWORD *)(v12 + 32), *(_QWORD *)(a1[6] + 112));
    v22[0] = a1;
    v14 = *a3;
    v17 = v13;
    v22[1] = &v19;
    v22[3] = sub_2AAE880;
    v22[2] = sub_2AA7C90;
    v18 = sub_2BF1270(v22, a5);
    v20 = *(_QWORD *)(v19 + 48);
    if ( v20 )
      sub_2AAAFA0(&v20);
    v10 = sub_22077B0(0xA8u);
    if ( v10 )
    {
      v15 = v19;
      v21 = v20;
      if ( v20 )
      {
        v16 = v19;
        sub_2AAAFA0(&v21);
        v15 = v16;
      }
      sub_2AAFC70(v10, 34, v15, v14, v17, v12, &v21);
      sub_9C6650(&v21);
      *(_QWORD *)v10 = &unk_4A24A10;
      *(_QWORD *)(v10 + 96) = &unk_4A24A98;
      *(_QWORD *)(v10 + 40) = &unk_4A24A60;
      *(_BYTE *)(v10 + 160) = v18;
    }
    sub_9C6650(&v20);
    sub_A17130((__int64)v22);
  }
  return v10;
}

// Function: sub_2A41DE0
// Address: 0x2a41de0
//
__int64 __fastcall sub_2A41DE0(
        __int64 **a1,
        const void *a2,
        size_t a3,
        unsigned __int8 a4,
        int a5,
        int a6,
        char *src,
        size_t n)
{
  __int64 *v9; // rax
  __int64 **v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // rbx
  __int64 *v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v19; // [rsp+8h] [rbp-88h]
  _QWORD *v20; // [rsp+10h] [rbp-80h]
  __int64 v21; // [rsp+10h] [rbp-80h]
  __int64 v23; // [rsp+28h] [rbp-68h] BYREF
  __int64 v24[4]; // [rsp+30h] [rbp-60h] BYREF
  char v25; // [rsp+50h] [rbp-40h]
  char v26; // [rsp+51h] [rbp-3Fh]

  v9 = (__int64 *)sub_BCD140(*a1, 8u);
  v10 = (__int64 **)sub_BCD420(v9, n);
  v11 = sub_AC9630(src, n, v10);
  v12 = *(_QWORD **)(v11 + 8);
  v19 = v11;
  v24[0] = (__int64)"llvm.embedded.object";
  v20 = v12;
  v26 = 1;
  v25 = 3;
  BYTE4(v23) = 0;
  v13 = sub_BD2C40(88, unk_3F0FAE8);
  v14 = (__int64)v13;
  if ( v13 )
    sub_B30000((__int64)v13, (__int64)a1, v20, 1, 8, v19, (__int64)v24, 0, 0, v23, 0);
  sub_B31A00(v14, (__int64)a2, a3);
  sub_B2F770(v14, a4);
  v15 = *a1;
  v21 = sub_BA8E40((__int64)a1, "llvm.embedded.objects", 0x15u);
  v24[0] = (__int64)sub_B98A20(v14, (__int64)"llvm.embedded.objects");
  v24[1] = sub_B9B140(v15, a2, a3);
  v16 = sub_B9C770(v15, v24, (__int64 *)2, 0, 1);
  sub_B979A0(v21, v16);
  v17 = sub_B9C770(v15, 0, 0, 0, 1);
  sub_B99110(v14, 33, v17);
  v23 = v14;
  return sub_2A41DC0(a1, (unsigned __int64 *)&v23, 1);
}

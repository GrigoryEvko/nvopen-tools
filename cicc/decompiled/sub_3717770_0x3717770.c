// Function: sub_3717770
// Address: 0x3717770
//
__int64 __fastcall sub_3717770(
        __int64 **a1,
        unsigned __int16 a2,
        unsigned __int64 a3,
        char *a4,
        signed __int64 a5,
        __int64 a6,
        unsigned int a7,
        __int64 a8,
        unsigned __int64 a9)
{
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // r9
  const char *v15; // rax
  _QWORD *v16; // rdx
  _QWORD *v17; // rax
  unsigned __int64 v18; // r15
  _QWORD *v19; // rax
  __int64 *v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 **v23; // rax
  __int64 v25; // [rsp+8h] [rbp-C8h]
  _QWORD *v26; // [rsp+10h] [rbp-C0h]
  __int64 v27; // [rsp+10h] [rbp-C0h]
  __int64 v31; // [rsp+30h] [rbp-A0h]
  __int64 v32; // [rsp+38h] [rbp-98h]
  _QWORD *v33; // [rsp+48h] [rbp-88h] BYREF
  __int64 v34; // [rsp+50h] [rbp-80h] BYREF
  __int64 v35; // [rsp+58h] [rbp-78h]
  __int64 v36; // [rsp+60h] [rbp-70h]
  __int64 v37; // [rsp+68h] [rbp-68h]
  __int64 v38; // [rsp+70h] [rbp-60h]
  __int64 v39; // [rsp+78h] [rbp-58h]
  __int64 v40; // [rsp+80h] [rbp-50h]
  __int64 v41; // [rsp+88h] [rbp-48h]
  __int64 v42; // [rsp+90h] [rbp-40h]

  v12 = sub_BCE3C0(*a1, 0);
  v13 = sub_BCB2E0(*a1);
  v31 = sub_BCB2D0(*a1);
  v32 = sub_BCB2C0(*a1);
  v14 = sub_AC9B20((__int64)*a1, a4, a5, 1);
  v25 = v14;
  v15 = "$offloading$entry_name";
  if ( (unsigned int)(*((_DWORD *)a1 + 66) - 42) >= 2 )
    v15 = ".offloading.entry_name";
  v16 = *(_QWORD **)(v14 + 8);
  v35 = 22;
  BYTE4(v33) = 0;
  v34 = (__int64)v15;
  v26 = v16;
  LOWORD(v38) = 261;
  v17 = sub_BD2C40(88, unk_3F0FAE8);
  v18 = (unsigned __int64)v17;
  if ( v17 )
    sub_B30000((__int64)v17, (__int64)a1, v26, 1, 7, v25, (__int64)&v34, 0, 0, (__int64)v33, 0);
  *(_BYTE *)(v18 + 32) = *(_BYTE *)(v18 + 32) & 0x3F | 0x80;
  sub_B31A00(v18, (__int64)".llvm.rodata.offloading", 23);
  sub_B2F770(v18, 0);
  v27 = sub_BA8E40((__int64)a1, "llvm.offloading.symbols", 0x17u);
  v19 = sub_B98A20(v18, (__int64)"llvm.offloading.symbols");
  v20 = *a1;
  v33 = v19;
  v21 = sub_B9C770(v20, (__int64 *)&v33, (__int64 *)1, 0, 1);
  sub_B979A0(v27, v21);
  v34 = sub_AD6530(v13, v21);
  v35 = sub_AD64C0(v32, 1, 0);
  v36 = sub_AD64C0(v32, a2, 0);
  v37 = sub_AD64C0(v31, a7, 0);
  v38 = sub_ADB060(a3, v12);
  v39 = sub_ADB060(v18, v12);
  v40 = sub_AD64C0(v13, a6, 0);
  v41 = sub_AD64C0(v13, a8, 0);
  if ( a9 )
    v22 = sub_ADB060(a9, v12);
  else
    v22 = sub_AD6530(v12, a8);
  v42 = v22;
  v23 = (__int64 **)sub_3717640(a1);
  return sub_AD24A0(v23, &v34, 9);
}

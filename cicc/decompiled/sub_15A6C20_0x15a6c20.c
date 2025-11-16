// Function: sub_15A6C20
// Address: 0x15a6c20
//
__int64 __fastcall sub_15A6C20(
        __int64 *a1,
        int a2,
        int a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __int64 a7,
        __int64 a8,
        int a9,
        __int64 a10,
        __int64 a11,
        int a12,
        __int64 a13,
        unsigned __int8 a14,
        unsigned __int8 a15,
        unsigned __int8 a16)
{
  int v16; // r15d
  __int64 v18; // r12
  __int64 v20; // r11
  __int64 v21; // r10
  __int64 v22; // r13
  __int64 v23; // rax
  int v24; // r9d
  int v25; // eax
  int v26; // ecx
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  int v34; // [rsp+10h] [rbp-50h]

  v16 = a6;
  v18 = 0;
  v20 = a7;
  v21 = a8;
  v22 = a1[1];
  if ( a11 )
  {
    v32 = a5;
    v23 = sub_161FF10(v22, a10, a11);
    v21 = a8;
    v20 = a7;
    a5 = v32;
    v18 = v23;
  }
  v24 = 0;
  if ( v21 )
  {
    v33 = a5;
    v25 = sub_161FF10(v22, v20, v21);
    a5 = v33;
    v24 = v25;
  }
  v26 = 0;
  if ( a5 )
  {
    v34 = v24;
    v27 = sub_161FF10(v22, a4, a5);
    v24 = v34;
    v26 = v27;
  }
  v28 = sub_15B0DC0(v22, a2, a3, v26, v16, v24, a9, v18, a12, 0, 0, 0, 0, 0, a13, a14, a15, a16, 1);
  v29 = *a1;
  a1[2] = v28;
  v30 = sub_1632440(v29, "llvm.dbg.cu", 11);
  sub_1623CA0(v30, a1[2]);
  sub_15A6B80((__int64)a1, a1[2]);
  return a1[2];
}

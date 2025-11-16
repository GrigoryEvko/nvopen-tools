// Function: sub_ADDEF0
// Address: 0xaddef0
//
__int64 __fastcall sub_ADDEF0(
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
        int a16,
        unsigned __int8 a17,
        __int64 a18,
        __int64 a19,
        __int64 a20,
        __int64 a21)
{
  __int64 v23; // r8
  __int64 v24; // r10
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r12
  int v29; // r9d
  int v30; // ecx
  int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v36; // [rsp+0h] [rbp-80h]
  int v37; // [rsp+10h] [rbp-70h]
  int v40; // [rsp+30h] [rbp-50h]
  __int64 v41; // [rsp+40h] [rbp-40h]
  __int64 v42; // [rsp+48h] [rbp-38h]

  v23 = a11;
  v24 = a18;
  v25 = a1[1];
  v42 = 0;
  v40 = a6;
  if ( a21 )
  {
    v26 = sub_B9B140(v25, a20, a21);
    v24 = a18;
    v23 = a11;
    v42 = v26;
  }
  v41 = 0;
  if ( a19 )
  {
    v36 = v23;
    v27 = sub_B9B140(v25, v24, a19);
    v23 = v36;
    v41 = v27;
  }
  v28 = 0;
  if ( v23 )
    v28 = sub_B9B140(v25, a10, v23);
  v29 = 0;
  if ( a8 )
    v29 = sub_B9B140(v25, a7, a8);
  v30 = 0;
  if ( a5 )
  {
    v37 = v29;
    v31 = sub_B9B140(v25, a4, a5);
    v29 = v37;
    v30 = v31;
  }
  v32 = sub_AF30C0(v25, a2, a3, v30, v40, v29, a9, v28, a12, 0, 0, 0, 0, 0, a13, a14, a15, a16, a17, v41, v42, 1);
  v33 = *a1;
  a1[2] = v32;
  v34 = sub_BA8E40(v33, "llvm.dbg.cu", 11);
  sub_B979A0(v34, a1[2]);
  sub_ADDDC0((__int64)a1, a1[2]);
  return a1[2];
}

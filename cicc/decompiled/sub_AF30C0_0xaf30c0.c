// Function: sub_AF30C0
// Address: 0xaf30c0
//
__int64 __fastcall sub_AF30C0(
        int a1,
        int a2,
        __int64 a3,
        __int64 a4,
        char a5,
        __int64 a6,
        int a7,
        __int64 a8,
        int a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        __int64 a14,
        __int64 a15,
        char a16,
        unsigned __int8 a17,
        int a18,
        char a19,
        __int64 a20,
        __int64 a21,
        unsigned int a22)
{
  __int64 v22; // rax
  __int64 v23; // r12
  _QWORD v26[18]; // [rsp+20h] [rbp-90h] BYREF

  v26[3] = a8;
  v26[4] = a10;
  v26[0] = a3;
  v26[5] = a11;
  v26[1] = a4;
  v26[6] = a12;
  v26[2] = a6;
  v26[7] = a13;
  v26[8] = a14;
  v26[9] = a20;
  v26[10] = a21;
  v22 = sub_B97910(48, 11, a22);
  v23 = v22;
  if ( v22 )
    sub_AF3020(v22, a1, a22, a2, a5, a7, a9, a15, a16, a17, a18, a19, (__int64)v26, 11);
  if ( !a22 )
    BUG();
  if ( a22 == 1 )
    sub_B95A20(v23);
  return v23;
}

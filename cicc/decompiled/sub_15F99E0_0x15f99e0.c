// Function: sub_15F99E0
// Address: 0x15f99e0
//
__int64 __fastcall sub_15F99E0(
        __int64 a1,
        __int64 a2,
        __int64 **a3,
        __int64 a4,
        __int16 a5,
        unsigned int a6,
        char a7,
        __int64 a8)
{
  __int64 v8; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  _QWORD v19[14]; // [rsp+40h] [rbp-70h] BYREF

  v8 = a1 - 72;
  v13 = sub_16498A0(a3);
  v14 = sub_1643320(v13);
  v15 = **a3;
  v19[0] = *a3;
  v19[1] = v14;
  v16 = sub_1645600(v15, v19, 2, 0);
  sub_15F1EA0(a1, v16, 34, v8, 3, a8);
  return sub_15F9860(a1, a2, (__int64)a3, a4, a5, a6, a7);
}

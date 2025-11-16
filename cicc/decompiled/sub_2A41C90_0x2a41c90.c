// Function: sub_2A41C90
// Address: 0x2a41c90
//
__int64 *__fastcall sub_2A41C90(
        __int64 *a1,
        __int64 **a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        const void *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        void (__fastcall *a11)(__int64, __int64, __int64, __int64),
        __int64 a12,
        __int64 a13,
        unsigned __int64 a14,
        char a15)
{
  _BYTE *v17; // rax
  _BYTE *v18; // r13
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v22; // r14
  __int64 v23; // r13
  __int64 v24; // rbx
  __int64 v25; // [rsp+0h] [rbp-80h]
  __int64 v28; // [rsp+30h] [rbp-50h] BYREF
  __int64 v29; // [rsp+38h] [rbp-48h]
  __int64 v30; // [rsp+40h] [rbp-40h]

  v17 = sub_BA8CB0((__int64)a2, a3, a4);
  if ( v17
    && ((v18 = v17, !*((_QWORD *)v17 + 13)) || (v25 = **(_QWORD **)(*((_QWORD *)v17 + 3) + 16LL), v25 == sub_BCB120(*a2))) )
  {
    v19 = sub_2A3F100(a2, a5, a6, a7, a8, a15);
    *a1 = (__int64)v18;
    a1[1] = v19;
    a1[2] = v20;
  }
  else
  {
    sub_2A41510(&v28, a2, a3, a4, a5, a6, a7, a8, a9, a10, a13, a14, a15);
    v22 = v28;
    v23 = v29;
    v24 = v30;
    a11(a12, v28, v29, v30);
    *a1 = v22;
    a1[1] = v23;
    a1[2] = v24;
  }
  return a1;
}

// Function: sub_F565E0
// Address: 0xf565e0
//
unsigned __int8 *__fastcall sub_F565E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v6; // r13
  __int64 v7; // rbx
  _QWORD *v8; // rax
  __int64 v9; // r15
  __int64 v10; // rbx
  _QWORD v12[8]; // [rsp+0h] [rbp-40h] BYREF

  v6 = (unsigned __int8 *)sub_F560C0((unsigned __int8 *)a1, a2, a3, a4, a5, a6);
  sub_BD6B90(v6, (unsigned __int8 *)a1);
  sub_B44220(v6, a1 + 24, 0);
  sub_BD84D0(a1, (__int64)v6);
  v7 = *(_QWORD *)(a1 - 96);
  v8 = sub_BD2C40(72, 1u);
  if ( v8 )
    sub_B4C8F0((__int64)v8, v7, 1u, a1 + 24, 0);
  v9 = *(_QWORD *)(a1 + 40);
  v10 = *(_QWORD *)(a1 - 64);
  sub_AA5980(v10, v9, 0);
  sub_B43D60((_QWORD *)a1);
  if ( a2 )
  {
    v12[0] = v9;
    v12[1] = v10 | 4;
    sub_FFB3D0(a2, v12, 1);
  }
  return v6;
}

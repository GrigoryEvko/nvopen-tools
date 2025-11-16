// Function: sub_C98ED0
// Address: 0xc98ed0
//
int __fastcall sub_C98ED0(int a1, __int64 a2, __int64 a3, int a4, int a5)
{
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  _BYTE *v10; // r13
  _QWORD *v11; // rax
  _QWORD *v12; // rbx
  _QWORD *v13; // rax
  __int64 v15; // [rsp+8h] [rbp-38h]

  v7 = a5;
  sub_C94E20((__int64)&qword_4F84F00);
  v8 = sub_C80C60(a2, a3, 0);
  v15 = v9;
  v10 = (_BYTE *)v8;
  v11 = (_QWORD *)sub_22077B0(16672);
  v12 = v11;
  if ( v11 )
  {
    v11[2068] = 0;
    *v11 = v11 + 2;
    v11[1] = 0x1000000000LL;
    v11[18] = v11 + 20;
    v11[19] = 0x8000000000LL;
    v11[2069] = 0;
    v11[2070] = 0x1800000000LL;
    v11[2071] = sub_220F850(16672);
    v12[2072] = sub_220F880() - 1000000 * v7;
    v12[2073] = v12 + 2075;
    sub_C95DE0(v12 + 2073, v10, (__int64)&v10[v15]);
    *((_DWORD *)v12 + 4154) = a4;
    v12[2078] = v12 + 2081;
    v12[2079] = 0;
    v12[2080] = 0;
    v12[2081] = 0;
    *((_BYTE *)v12 + 16656) = 0;
    *((_DWORD *)v12 + 4165) = a1;
    *((_BYTE *)v12 + 16664) = 0;
    sub_C95A60((__int64)(v12 + 2078));
  }
  v13 = (_QWORD *)sub_CEECD0(8, 8);
  *v13 = v12;
  return sub_C94E10((__int64)&qword_4F84F00, v13);
}

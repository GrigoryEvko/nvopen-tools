// Function: sub_C98D50
// Address: 0xc98d50
//
int __fastcall sub_C98D50(int a1, __int64 a2, __int64 a3, char a4)
{
  _BYTE *v6; // r13
  __int64 v7; // rdx
  __int64 v8; // r12
  _QWORD *v9; // rax
  _QWORD *v10; // rbx
  __pid_t v11; // eax
  __int64 v12; // rax
  _QWORD *v13; // rax

  sub_C94E20((__int64)&qword_4F84F00);
  v6 = (_BYTE *)sub_C80C60(a2, a3, 0);
  v8 = v7;
  v9 = (_QWORD *)sub_22077B0(16672);
  v10 = v9;
  if ( v9 )
  {
    v9[2068] = 0;
    *v9 = v9 + 2;
    v9[1] = 0x1000000000LL;
    v9[18] = v9 + 20;
    v9[19] = 0x8000000000LL;
    v9[2069] = 0;
    v9[2070] = 0x1800000000LL;
    v9[2071] = sub_220F850(16672);
    v10[2072] = sub_220F880();
    v10[2073] = v10 + 2075;
    sub_C95DE0(v10 + 2073, v6, (__int64)&v6[v8]);
    v11 = j__getpid();
    v10[2079] = 0;
    *((_DWORD *)v10 + 4154) = v11;
    v10[2078] = v10 + 2081;
    v10[2080] = 0;
    v12 = sub_C959E0();
    *((_BYTE *)v10 + 16656) = 1;
    v10[2081] = v12;
    *((_DWORD *)v10 + 4165) = a1;
    *((_BYTE *)v10 + 16664) = a4;
    sub_C95A60((__int64)(v10 + 2078));
  }
  v13 = (_QWORD *)sub_CEECD0(8, 8);
  *v13 = v10;
  return sub_C94E10((__int64)&qword_4F84F00, v13);
}

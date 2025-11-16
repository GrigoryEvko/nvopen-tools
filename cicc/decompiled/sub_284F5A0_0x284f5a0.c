// Function: sub_284F5A0
// Address: 0x284f5a0
//
unsigned __int64 __fastcall sub_284F5A0(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rax
  _QWORD *v11; // rax
  unsigned __int64 v12; // rax
  __int64 v14; // [rsp+0h] [rbp-40h]

  v14 = a1 + 48 * a2;
  if ( a1 != v14 )
  {
    v7 = a1;
    v8 = 0;
    while ( 1 )
    {
      v9 = v8;
      v10 = sub_1055AA0(a3, v7, a5);
      v11 = sub_DC5890(a5, v10, a4);
      v12 = sub_1055B50((unsigned __int64)v11, v7, a5, 1);
      v8 = v12;
      if ( !v12 || v9 && v12 != v9 )
        break;
      v7 += 48;
      if ( v14 == v7 )
        return v8;
    }
  }
  return 0;
}

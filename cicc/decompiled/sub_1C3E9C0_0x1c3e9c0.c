// Function: sub_1C3E9C0
// Address: 0x1c3e9c0
//
int __fastcall sub_1C3E9C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9

  if ( !qword_4FBA5B0 )
    sub_16C1EA0((__int64)&qword_4FBA5B0, (__int64 (*)(void))sub_1C3E6D0, (__int64)sub_1C3E470, a4, a5, a6);
  v6 = sub_16D40F0(qword_4FBA5B0);
  v7 = (__int64)v6;
  if ( v6 )
  {
    v8 = v6[1];
    if ( v8 )
    {
      v9 = sub_2207820(v8 + 1);
      *a1 = v9;
      v10 = v9;
      sub_2241570(v7, v9, v8, 0);
      *(_BYTE *)(*a1 + v8) = 0;
      LODWORD(v6) = sub_1C3E900(v7, v10, v11, v12, v13, v14);
    }
  }
  return (int)v6;
}

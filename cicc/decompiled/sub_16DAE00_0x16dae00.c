// Function: sub_16DAE00
// Address: 0x16dae00
//
int __fastcall sub_16DAE00(int a1, __int64 a2, __int64 a3, int a4, int a5)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  _BYTE *v10; // r13
  _QWORD *v11; // rax
  _QWORD *v12; // r14
  _QWORD *v13; // rax
  __int64 v15; // [rsp+8h] [rbp-38h]

  v6 = a4;
  v7 = a5;
  sub_16D40F0((__int64)&qword_4FA1650);
  v8 = sub_16C40A0(a2, a3, 2);
  v15 = v9;
  v10 = (_BYTE *)v8;
  v11 = (_QWORD *)sub_22077B0(11672);
  v12 = v11;
  if ( v11 )
  {
    v11[1444] = 0;
    *v11 = v11 + 2;
    v11[1] = 0x1000000000LL;
    v11[162] = v11 + 164;
    v11[163] = 0x8000000000LL;
    v11[1445] = 0;
    v11[1446] = 0x1800000000LL;
    v11[1448] = sub_220F850(11672);
    v12[1449] = sub_220F880() - 1000000 * v7;
    v12[1450] = v12 + 1452;
    if ( v10 )
    {
      sub_16D9940(v12 + 1450, v10, (__int64)&v10[v15]);
    }
    else
    {
      v12[1451] = 0;
      *((_BYTE *)v12 + 11616) = 0;
    }
    v12[1454] = v6;
    v12[1455] = v12 + 1457;
    v12[1456] = 0;
    v12[1457] = 0;
    *((_DWORD *)v12 + 2916) = a1;
    sub_16D5D60((__int64)(v12 + 1455));
  }
  v13 = (_QWORD *)sub_1C42D70(8, 8);
  *v13 = v12;
  return sub_16D40E0((__int64)&qword_4FA1650, v13);
}

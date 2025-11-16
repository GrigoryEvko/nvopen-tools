// Function: sub_3382930
// Address: 0x3382930
//
void __fastcall sub_3382930(__int64 a1, __int64 a2, _QWORD *a3, _DWORD **a4, int a5)
{
  __int64 **v9; // rax
  __int64 v10; // rax
  _DWORD *v11; // rsi
  __int64 v12; // rbx
  __int64 v13; // [rsp+0h] [rbp-40h] BYREF
  _DWORD *v14[7]; // [rsp+8h] [rbp-38h] BYREF

  v9 = (__int64 **)sub_BCB2A0(*(_QWORD **)(a1 + 1024));
  v13 = sub_ACADE0(v9);
  v10 = sub_B0D220(a3);
  v11 = *a4;
  v12 = v10;
  v14[0] = v11;
  if ( v11 )
    sub_B96E90((__int64)v14, (__int64)v11, 1);
  sub_3380DB0(a1, &v13, 1, a2, v12, v14, a5, 0);
  if ( v14[0] )
    sub_B91220((__int64)v14, (__int64)v14[0]);
}

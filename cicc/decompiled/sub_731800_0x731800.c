// Function: sub_731800
// Address: 0x731800
//
__int64 __fastcall sub_731800(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD v7[10]; // [rsp+0h] [rbp-F0h] BYREF
  unsigned int v8; // [rsp+50h] [rbp-A0h]
  int v9; // [rsp+5Ch] [rbp-94h]
  int v10; // [rsp+78h] [rbp-78h]

  sub_76C7C0(v7, a2, a3, a4, a5, a6);
  v7[0] = sub_73CC20;
  v7[4] = sub_728160;
  v7[2] = sub_728BE0;
  v9 = 1;
  sub_76D3C0(a1, v7);
  if ( a2 )
    *a2 = v10;
  return v8;
}

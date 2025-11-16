// Function: sub_731890
// Address: 0x731890
//
__int64 __fastcall sub_731890(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD v8[10]; // [rsp+0h] [rbp-F0h] BYREF
  unsigned int v9; // [rsp+50h] [rbp-A0h]
  int v10; // [rsp+5Ch] [rbp-94h]
  int v11; // [rsp+78h] [rbp-78h]
  int v12; // [rsp+7Ch] [rbp-74h]

  sub_76C7C0(v8, a2, a3, a4, a5, a6);
  v12 = a2;
  v10 = 1;
  v8[0] = sub_73CC20;
  v8[4] = sub_728160;
  v8[2] = sub_728BE0;
  sub_76D400(a1, v8);
  if ( a3 )
    *a3 = v11;
  return v9;
}

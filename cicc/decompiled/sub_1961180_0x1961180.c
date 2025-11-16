// Function: sub_1961180
// Address: 0x1961180
//
__int64 __fastcall sub_1961180(__int64 a1, __int64 a2, __int64 a3, char *a4, __int64 *a5)
{
  __int64 v8; // r13
  unsigned __int64 v9; // rax
  __int64 v11; // rsi
  __int64 v12; // [rsp+18h] [rbp-38h]

  v8 = sub_13FC520(a3);
  sub_1960E20(a5, a1);
  if ( *(__int16 *)(a1 + 18) < 0 && !(unsigned __int8)sub_1437020(a1, a2, a3, a4) )
    sub_1624960(a1, 0, 0);
  v9 = sub_157EBA0(v8);
  sub_15F22F0((_QWORD *)a1, v9);
  if ( *(_BYTE *)(a1 + 16) != 78 )
  {
    v11 = *(_QWORD *)(a1 + 48);
    v12 = 0;
    if ( v11 )
    {
      sub_161E7C0(a1 + 48, v11);
      *(_QWORD *)(a1 + 48) = v12;
    }
  }
  return 1;
}

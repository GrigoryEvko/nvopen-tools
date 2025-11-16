// Function: sub_731B40
// Address: 0x731b40
//
__int64 __fastcall sub_731B40(__int64 a1, _QWORD *a2, __int64 a3, _QWORD a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  unsigned int v7; // r13d
  _QWORD v9[2]; // [rsp+0h] [rbp-100h] BYREF
  _QWORD v10[10]; // [rsp+10h] [rbp-F0h] BYREF
  unsigned int v11; // [rsp+60h] [rbp-A0h]

  v9[0] = 0;
  v9[1] = 0;
  v6 = word_4D04898;
  if ( !word_4D04898 || !dword_4D048AC || (a2 = v9, v7 = 0, !(unsigned int)sub_7A2E10(a1, v9)) )
  {
    sub_76C7C0(v10, a2, a3, v6, a5, a6);
    v10[0] = sub_72B1B0;
    v10[4] = sub_7288B0;
    v10[2] = sub_728200;
    if ( dword_4D048B8 )
      sub_76CDC0(a1);
    v7 = v11;
  }
  sub_67E3D0(v9);
  return v7;
}

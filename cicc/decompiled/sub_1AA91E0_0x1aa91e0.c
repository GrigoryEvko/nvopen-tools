// Function: sub_1AA91E0
// Address: 0x1aa91e0
//
__int64 __fastcall sub_1AA91E0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // r15d
  unsigned __int64 v7; // r14
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  _QWORD v12[2]; // [rsp+10h] [rbp-50h] BYREF
  int v13; // [rsp+20h] [rbp-40h]

  v6 = sub_137DFF0((__int64)a1, (__int64)a2);
  v12[1] = a4;
  v7 = sub_157EBA0((__int64)a1);
  v13 = (int)&loc_1010000;
  v12[0] = a3;
  if ( sub_1AAC5F0(v7, v6, v12) )
    return sub_15F4DF0(v7, v6);
  if ( sub_157F0B0((__int64)a2) )
  {
    v9 = a2[6];
    if ( v9 )
      v9 -= 24;
    return sub_1AA8CA0(a2, v9, a3, a4);
  }
  else
  {
    v10 = sub_157EBA0((__int64)a1);
    return sub_1AA8CA0(a1, v10, a3, a4);
  }
}

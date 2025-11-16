// Function: sub_1632210
// Address: 0x1632210
//
__int64 __fastcall sub_1632210(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 ***v5; // rax
  __int64 ***v6; // r12
  __int64 **v7; // r13
  bool v8; // zf
  __int64 **v9; // rax
  __int64 **v10; // rsi
  _QWORD v12[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD *v13; // [rsp+10h] [rbp-40h] BYREF
  __int16 v14; // [rsp+20h] [rbp-30h]

  v12[0] = a2;
  v12[1] = a3;
  v5 = (__int64 ***)sub_1632000(a1, a2, a3);
  if ( v5 && (v6 = v5, *((_BYTE *)v5 + 16) == 3) )
  {
    v7 = *v5;
    v8 = *((_BYTE *)*v5 + 8) == 16;
    v9 = *v5;
    if ( v8 )
      v9 = (__int64 **)*v7[2];
    v10 = (__int64 **)sub_1646BA0(a4, *((_DWORD *)v9 + 2) >> 8);
    if ( v10 != v7 )
      return sub_15A4510(v6, v10, 0);
  }
  else
  {
    v14 = 261;
    v13 = v12;
    v6 = (__int64 ***)sub_1648A60(88, 1);
    if ( v6 )
      sub_15E51E0((__int64)v6, a1, a4, 0, 0, 0, (__int64)&v13, 0, 0, 0, 0);
  }
  return (__int64)v6;
}

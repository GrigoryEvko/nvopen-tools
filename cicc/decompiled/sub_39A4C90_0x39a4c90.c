// Function: sub_39A4C90
// Address: 0x39a4c90
//
__int64 __fastcall sub_39A4C90(__int64 *a1, __int64 a2, __int16 a3, __int64 **a4)
{
  __int64 v6; // rsi
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  int v9; // ecx
  __int16 v10; // ax
  __int64 **v12; // [rsp+8h] [rbp-38h] BYREF
  __int64 v13[6]; // [rsp+10h] [rbp-30h] BYREF

  v6 = a1[24];
  v12 = a4;
  sub_3982A70(a4, v6);
  v7 = (_BYTE *)a1[33];
  if ( v7 == (_BYTE *)a1[34] )
  {
    sub_39A4B00((__int64)(a1 + 32), v7, &v12);
    v8 = (__int64)v12;
  }
  else
  {
    v8 = (__int64)v12;
    if ( v7 )
    {
      *(_QWORD *)v7 = v12;
      v7 = (_BYTE *)a1[33];
    }
    a1[33] = (__int64)(v7 + 8);
  }
  v9 = *(_DWORD *)(v8 + 8);
  v10 = 10;
  if ( (v9 & 0xFFFFFF00) != 0 )
    v10 = ((v9 & 0xFFFF0000) != 0) + 3;
  v13[1] = v8;
  WORD2(v13[0]) = a3;
  LODWORD(v13[0]) = 7;
  HIWORD(v13[0]) = v10;
  return sub_39A31C0((__int64 *)(a2 + 8), a1 + 11, v13);
}

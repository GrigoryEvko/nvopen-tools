// Function: sub_39A4520
// Address: 0x39a4520
//
__int64 __fastcall sub_39A4520(__int64 *a1, __int64 a2, __int16 a3, __int64 **a4)
{
  __int64 v6; // rsi
  _BYTE *v7; // rsi
  __int64 v8; // r14
  __int64 *v9; // r12
  unsigned __int16 v10; // r8
  __int16 v11; // ax
  int v12; // edx
  __int64 v14; // [rsp+8h] [rbp-38h] BYREF
  __int64 v15[6]; // [rsp+10h] [rbp-30h] BYREF

  v6 = a1[24];
  v14 = (__int64)a4;
  sub_3982A00(a4, v6);
  v7 = (_BYTE *)a1[36];
  if ( v7 == (_BYTE *)a1[37] )
  {
    sub_39A4390((__int64)(a1 + 35), v7, &v14);
    v8 = v14;
  }
  else
  {
    v8 = v14;
    if ( v7 )
    {
      *(_QWORD *)v7 = v14;
      v7 = (_BYTE *)a1[36];
    }
    a1[36] = (__int64)(v7 + 8);
  }
  v9 = (__int64 *)(a2 + 8);
  v10 = sub_398C0A0(a1[25]);
  v11 = 24;
  if ( v10 <= 3u )
  {
    v12 = *(_DWORD *)(v8 + 8);
    v11 = 10;
    if ( (v12 & 0xFFFFFF00) != 0 )
      v11 = ((v12 & 0xFFFF0000) != 0) + 3;
  }
  HIWORD(v15[0]) = v11;
  WORD2(v15[0]) = a3;
  LODWORD(v15[0]) = 8;
  v15[1] = v14;
  return sub_39A31C0(v9, a1 + 11, v15);
}

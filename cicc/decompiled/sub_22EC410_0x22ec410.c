// Function: sub_22EC410
// Address: 0x22ec410
//
void __fastcall sub_22EC410(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  _QWORD *v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // [rsp+10h] [rbp-50h] BYREF
  __int64 v12; // [rsp+18h] [rbp-48h]
  _BYTE v13[64]; // [rsp+20h] [rbp-40h] BYREF

  sub_22EC3A0(a1, a2, a3, a4, 0);
  v7 = *(_BYTE **)(a4 + 168);
  if ( v7 )
  {
    v8 = (__int64)&v7[*(_QWORD *)(a4 + 176)];
    v11 = v13;
    sub_22EB340((__int64 *)&v11, v7, v8);
    v9 = v11;
    v10 = v12;
  }
  else
  {
    v12 = 0;
    v10 = 0;
    v11 = v13;
    v9 = v13;
    v13[0] = 0;
  }
  sub_22EC090(a1, a4, a2, a3, (__int64)v9, v10, (__int64)"Module", 6);
  if ( v11 != (_QWORD *)v13 )
    j_j___libc_free_0((unsigned __int64)v11);
}

// Function: sub_1C007A0
// Address: 0x1c007a0
//
void __fastcall sub_1C007A0(__int64 a1, __int64 a2, char a3, char a4, char a5, char a6)
{
  __int64 v8; // rax
  _BYTE *v9; // rsi
  __int64 v12; // [rsp+10h] [rbp-150h] BYREF
  _BYTE *v13; // [rsp+18h] [rbp-148h]
  _BYTE *v14; // [rsp+20h] [rbp-140h]
  __int64 v15; // [rsp+30h] [rbp-130h] BYREF
  _QWORD *v16; // [rsp+38h] [rbp-128h]
  _QWORD *v17; // [rsp+40h] [rbp-120h]
  __int64 v18; // [rsp+48h] [rbp-118h]
  int v19; // [rsp+50h] [rbp-110h]
  _QWORD v20[8]; // [rsp+58h] [rbp-108h] BYREF
  __int64 v21; // [rsp+98h] [rbp-C8h] BYREF
  __int64 v22; // [rsp+A0h] [rbp-C0h]
  __int64 v23; // [rsp+A8h] [rbp-B8h]
  _QWORD v24[22]; // [rsp+B0h] [rbp-B0h] BYREF

  v8 = *(_QWORD *)(a2 + 80);
  v18 = 0x100000008LL;
  if ( v8 )
    v8 -= 24;
  v12 = 0;
  v13 = 0;
  v20[0] = v8;
  v24[0] = v8;
  v14 = 0;
  v16 = v20;
  v17 = v20;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v19 = 0;
  v15 = 1;
  LOBYTE(v24[3]) = 0;
  sub_144A690(&v21, (__int64)v24);
  memset(v24, 0, 0x80u);
  LODWORD(v24[3]) = 8;
  v24[1] = &v24[5];
  v24[2] = &v24[5];
  while ( v21 != v22 )
  {
    v9 = v13;
    if ( v13 == v14 )
    {
      sub_1292090((__int64)&v12, v13, (_QWORD *)(v22 - 32));
    }
    else
    {
      if ( v13 )
      {
        *(_QWORD *)v13 = *(_QWORD *)(v22 - 32);
        v9 = v13;
      }
      v13 = v9 + 8;
    }
    sub_17D3A30((__int64)&v15);
  }
  if ( v21 )
    j_j___libc_free_0(v21, v23 - v21);
  if ( v17 != v16 )
    _libc_free((unsigned __int64)v17);
  sub_1BFE7F0(a1, &v12, a3, a4, a5, a6);
  if ( v12 )
    j_j___libc_free_0(v12, &v14[-v12]);
}

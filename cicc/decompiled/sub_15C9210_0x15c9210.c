// Function: sub_15C9210
// Address: 0x15c9210
//
__int64 __fastcall sub_15C9210(__int64 a1, __int64 a2)
{
  bool v2; // zf
  int v3; // eax
  int v4; // edx
  int v6; // [rsp+8h] [rbp-E8h] BYREF
  int v7; // [rsp+Ch] [rbp-E4h] BYREF
  _QWORD v8[2]; // [rsp+10h] [rbp-E0h] BYREF
  _QWORD v9[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int16 v10; // [rsp+30h] [rbp-C0h]
  __int64 v11; // [rsp+40h] [rbp-B0h]
  _QWORD v12[2]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v13; // [rsp+70h] [rbp-80h]
  _QWORD v14[2]; // [rsp+80h] [rbp-70h] BYREF
  __int16 v15; // [rsp+90h] [rbp-60h]
  __int64 v16; // [rsp+A0h] [rbp-50h]
  _QWORD v17[2]; // [rsp+C0h] [rbp-30h] BYREF
  __int16 v18; // [rsp+D0h] [rbp-20h]

  v2 = *(_QWORD *)(a2 + 32) == 0;
  v8[0] = "<unknown>";
  v8[1] = 9;
  v6 = 0;
  v7 = 0;
  if ( v2 )
  {
    v3 = 0;
    v4 = 0;
  }
  else
  {
    sub_15C91F0(a2, v8, &v6, &v7);
    v4 = v7;
    v3 = v6;
  }
  LODWORD(v16) = v4;
  LODWORD(v11) = v3;
  v10 = 773;
  v12[0] = v9;
  v9[1] = ":";
  v14[1] = ":";
  v12[1] = v11;
  v17[0] = v14;
  v13 = 2306;
  v18 = 2306;
  v9[0] = v8;
  v17[1] = v16;
  v14[0] = v12;
  v15 = 770;
  sub_16E2FC0(a1, v17);
  return a1;
}

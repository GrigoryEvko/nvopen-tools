// Function: sub_1DDC610
// Address: 0x1ddc610
//
__int64 *__fastcall sub_1DDC610(__int64 *a1, __int64 a2)
{
  int v2; // eax
  bool v3; // zf
  __int64 v4; // rdx
  _QWORD v6[2]; // [rsp+0h] [rbp-C0h] BYREF
  _QWORD v7[2]; // [rsp+10h] [rbp-B0h] BYREF
  __int16 v8; // [rsp+20h] [rbp-A0h]
  _QWORD v9[2]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v10; // [rsp+40h] [rbp-80h]
  _QWORD v11[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v12; // [rsp+60h] [rbp-60h]
  char *v13; // [rsp+70h] [rbp-50h]
  char v14; // [rsp+80h] [rbp-40h]
  char v15; // [rsp+81h] [rbp-3Fh]
  _QWORD v16[2]; // [rsp+90h] [rbp-30h] BYREF
  __int64 v17; // [rsp+A0h] [rbp-20h]

  v2 = *(_DWORD *)(a2 + 48);
  v3 = *(_QWORD *)(a2 + 40) == 0;
  v8 = 2563;
  LODWORD(v16[0]) = v2;
  v7[0] = "BB";
  v7[1] = v16[0];
  if ( v3 )
  {
    sub_16E2FC0(a1, (__int64)v7);
  }
  else
  {
    v15 = 1;
    v13 = "]";
    v14 = 3;
    v6[0] = sub_1DD6290(a2);
    v6[1] = v4;
    v9[1] = "[";
    v9[0] = v7;
    v10 = 770;
    v11[0] = v9;
    v11[1] = v6;
    LOWORD(v12) = 1282;
    v16[1] = "]";
    v16[0] = v11;
    LOWORD(v17) = 770;
    sub_16E2FC0(a1, (__int64)v16);
  }
  return a1;
}

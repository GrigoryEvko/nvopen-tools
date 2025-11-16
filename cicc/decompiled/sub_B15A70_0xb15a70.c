// Function: sub_B15A70
// Address: 0xb15a70
//
__int64 __fastcall sub_B15A70(__int64 a1, __int64 a2)
{
  const char *v2; // rax
  bool v3; // zf
  int v4; // ecx
  int v5; // esi
  __int64 v6; // rdx
  int v8; // [rsp+8h] [rbp-E8h] BYREF
  int v9; // [rsp+Ch] [rbp-E4h] BYREF
  const char *v10; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v11; // [rsp+18h] [rbp-D8h]
  _QWORD v12[4]; // [rsp+20h] [rbp-D0h] BYREF
  __int16 v13; // [rsp+40h] [rbp-B0h]
  _QWORD *v14; // [rsp+50h] [rbp-A0h] BYREF
  int v15; // [rsp+60h] [rbp-90h]
  __int16 v16; // [rsp+70h] [rbp-80h]
  _QWORD v17[4]; // [rsp+80h] [rbp-70h] BYREF
  __int16 v18; // [rsp+A0h] [rbp-50h]
  _QWORD *v19; // [rsp+B0h] [rbp-40h] BYREF
  int v20; // [rsp+C0h] [rbp-30h]
  __int16 v21; // [rsp+D0h] [rbp-20h]

  v2 = "<unknown>";
  v3 = *(_QWORD *)(a2 + 24) == 0;
  v10 = "<unknown>";
  v11 = 9;
  v8 = 0;
  v9 = 0;
  if ( v3 )
  {
    v6 = 9;
    v5 = 0;
    v4 = 0;
  }
  else
  {
    sub_B15A30(a2, (__int64 *)&v10, &v8, &v9);
    v4 = v9;
    v5 = v8;
    v2 = v10;
    v6 = v11;
  }
  v20 = v4;
  v12[0] = v2;
  v12[1] = v6;
  v13 = 773;
  v12[2] = ":";
  v14 = v12;
  v15 = v5;
  v17[2] = ":";
  v19 = v17;
  v16 = 2306;
  v17[0] = &v14;
  v18 = 770;
  v21 = 2306;
  sub_CA0F50(a1, &v19);
  return a1;
}

// Function: sub_254C590
// Address: 0x254c590
//
__int64 *__fastcall sub_254C590(__int64 *a1, _BYTE *a2)
{
  char v2; // r15
  char v3; // bl
  __int64 v4; // rax
  _BYTE *v5; // rax
  __int64 v7; // rax
  char v8; // [rsp+Eh] [rbp-82h]
  char v9; // [rsp+Fh] [rbp-81h]
  char v10; // [rsp+1Eh] [rbp-72h] BYREF
  char v11; // [rsp+1Fh] [rbp-71h]
  _QWORD v12[3]; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v13; // [rsp+38h] [rbp-58h]
  _BYTE *v14; // [rsp+40h] [rbp-50h]
  __int64 v15; // [rsp+48h] [rbp-48h]
  __int64 *v16; // [rsp+50h] [rbp-40h]

  sub_253C590(a1, "AADenormalFPMath[");
  v12[1] = 0;
  v15 = 0x100000000LL;
  v12[2] = 0;
  v13 = 0;
  v14 = 0;
  v12[0] = &unk_49DD210;
  v16 = a1;
  sub_CB5980((__int64)v12, 0, 0, 0);
  v2 = a2[98];
  v3 = a2[99];
  v9 = a2[96];
  if ( v9 == -1 || (v8 = a2[97], v8 == -1) )
  {
    sub_904010((__int64)v12, "invalid");
    if ( v3 == -1 )
      goto LABEL_4;
  }
  else
  {
    v4 = sub_904010((__int64)v12, "denormal-fp-math=");
    v10 = v9;
    v11 = v8;
    sub_254C2A0(&v10, v4);
    if ( v3 == -1 )
      goto LABEL_4;
  }
  if ( v2 != -1 )
  {
    v7 = sub_904010((__int64)v12, " denormal-fp-math-f32=");
    v10 = v2;
    v11 = v3;
    sub_254C2A0(&v10, v7);
    v5 = v14;
    if ( (unsigned __int64)v14 < v13 )
      goto LABEL_5;
LABEL_10:
    sub_CB5D20((__int64)v12, 93);
    goto LABEL_6;
  }
LABEL_4:
  v5 = v14;
  if ( (unsigned __int64)v14 >= v13 )
    goto LABEL_10;
LABEL_5:
  v14 = v5 + 1;
  *v5 = 93;
LABEL_6:
  v12[0] = &unk_49DD210;
  sub_CB5840((__int64)v12);
  return a1;
}

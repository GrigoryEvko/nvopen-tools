// Function: sub_38E3370
// Address: 0x38e3370
//
__int64 __fastcall sub_38E3370(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  bool v6; // zf
  bool v7; // sf
  __int64 v8; // rdi
  __int64 v9; // rax
  unsigned __int8 v10; // al
  _QWORD v11[3]; // [rsp+0h] [rbp-100h] BYREF
  __int64 v12; // [rsp+18h] [rbp-E8h] BYREF
  _QWORD v13[2]; // [rsp+20h] [rbp-E0h] BYREF
  __int16 v14; // [rsp+30h] [rbp-D0h]
  _QWORD v15[2]; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v16; // [rsp+50h] [rbp-B0h]
  _QWORD v17[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v18; // [rsp+70h] [rbp-90h]
  _QWORD v19[2]; // [rsp+80h] [rbp-80h] BYREF
  __int16 v20; // [rsp+90h] [rbp-70h]
  _QWORD v21[2]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v22; // [rsp+B0h] [rbp-50h]
  _QWORD v23[2]; // [rsp+C0h] [rbp-40h] BYREF
  __int16 v24; // [rsp+D0h] [rbp-30h]

  v11[0] = a3;
  v11[1] = a4;
  v12 = 0;
  if ( (unsigned __int8)sub_3909470(a1, &v12) )
    return 1;
  v14 = 1283;
  v13[0] = "expected integer in '";
  v13[1] = v11;
  v15[0] = v13;
  v15[1] = "' directive";
  v16 = 770;
  if ( (unsigned __int8)sub_3909D40(a1, a2, v15) )
    return 1;
  v20 = 770;
  v6 = *a2 == 0;
  v7 = (__int64)*a2 < 0;
  v18 = 1283;
  v17[0] = "file number less than one in '";
  v17[1] = v11;
  v19[0] = v17;
  v19[1] = "' directive";
  if ( (unsigned __int8)sub_3909C80(a1, v7 | (unsigned __int8)v6, v12, v19) )
    return 1;
  v8 = *(_QWORD *)(a1 + 320);
  v21[1] = v11;
  v22 = 1283;
  v21[0] = "unassigned file number in '";
  v24 = 770;
  v23[0] = v21;
  v23[1] = "' directive";
  v9 = sub_38BE350(v8);
  v10 = sub_390FE20(v9, *a2);
  return sub_3909C80(a1, v10 ^ 1u, v12, v23);
}

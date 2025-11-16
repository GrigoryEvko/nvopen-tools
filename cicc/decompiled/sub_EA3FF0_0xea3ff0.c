// Function: sub_EA3FF0
// Address: 0xea3ff0
//
__int64 __fastcall sub_EA3FF0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  bool v9; // zf
  bool v10; // sf
  __int64 v11; // rsi
  __int64 v12; // rdi
  _QWORD *v13; // rax
  unsigned __int8 v14; // al
  __int64 v15; // [rsp+8h] [rbp-128h] BYREF
  const char *v16; // [rsp+10h] [rbp-120h] BYREF
  char v17; // [rsp+30h] [rbp-100h]
  char v18; // [rsp+31h] [rbp-FFh]
  _QWORD v19[4]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v20; // [rsp+60h] [rbp-D0h]
  _QWORD v21[4]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v22; // [rsp+90h] [rbp-A0h]
  _QWORD v23[4]; // [rsp+A0h] [rbp-90h] BYREF
  __int16 v24; // [rsp+C0h] [rbp-70h]
  _QWORD v25[4]; // [rsp+D0h] [rbp-60h] BYREF
  __int16 v26; // [rsp+F0h] [rbp-40h]

  v15 = 0;
  if ( (unsigned __int8)sub_ECD7C0(a1, &v15) )
    return 1;
  v18 = 1;
  v16 = "expected file number";
  v17 = 3;
  if ( (unsigned __int8)sub_ECE130(a1, a2, &v16) )
    return 1;
  v22 = 770;
  v9 = *a2 == 0;
  v10 = *a2 < 0;
  v20 = 1283;
  v11 = v10 | (unsigned __int8)v9;
  v19[0] = "file number less than one in '";
  v19[2] = a3;
  v19[3] = a4;
  v21[0] = v19;
  v21[2] = "' directive";
  if ( (unsigned __int8)sub_ECE070(a1, v11, v15, v21) )
    return 1;
  v12 = *(_QWORD *)(a1 + 224);
  v23[2] = a3;
  v24 = 1283;
  v23[0] = "unassigned file number in '";
  v26 = 770;
  v23[3] = a4;
  v25[0] = v23;
  v25[2] = "' directive";
  v13 = sub_E66210(v12, v11);
  v14 = sub_E5F770((__int64)v13, *a2);
  return sub_ECE070(a1, v14 ^ 1u, v15, v25);
}

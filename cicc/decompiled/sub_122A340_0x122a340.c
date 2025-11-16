// Function: sub_122A340
// Address: 0x122a340
//
__int64 __fastcall sub_122A340(__int64 a1, __int64 *a2, char a3)
{
  __int64 v4; // r13
  int v5; // eax
  char v6; // al
  unsigned __int64 v7; // r14
  unsigned int v8; // r15d
  __int64 *v9; // rdi
  __int64 v10; // rax
  int v12; // eax
  __int16 v13; // [rsp+1Eh] [rbp-E2h] BYREF
  __int64 v14; // [rsp+20h] [rbp-E0h] BYREF
  __int16 v15; // [rsp+28h] [rbp-D8h]
  __int64 v16; // [rsp+30h] [rbp-D0h] BYREF
  __int16 v17; // [rsp+38h] [rbp-C8h]
  __int64 v18; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v19; // [rsp+48h] [rbp-B8h]
  _QWORD v20[4]; // [rsp+50h] [rbp-B0h] BYREF
  _QWORD v21[4]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v22; // [rsp+90h] [rbp-70h]
  _QWORD v23[4]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v24; // [rsp+C0h] [rbp-40h]

  v13 = 0;
  v4 = a1 + 176;
  v20[0] = 48;
  v20[1] = 0;
  v20[2] = 0xFFFF;
  v14 = 0;
  v15 = 256;
  v16 = 0;
  v17 = 256;
  v18 = 0;
  v19 = 256;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    return 1;
  v5 = *(_DWORD *)(a1 + 240);
  if ( v5 == 13 )
  {
LABEL_8:
    v7 = *(_QWORD *)(a1 + 232);
    v8 = sub_120AFE0(a1, 13, "expected ')' here");
    if ( (_BYTE)v8 )
      return 1;
    if ( !(_BYTE)v19 )
    {
      v23[0] = "missing required field 'value'";
      v24 = 259;
      sub_11FD800(v4, v7, (__int64)v23, 1);
      return 1;
    }
    v9 = *(__int64 **)a1;
    if ( a3 )
      v10 = sub_B0B320(v9, v20[0], v14, v16, v13, v18, 1u, 1);
    else
      v10 = sub_B0B320(v9, v20[0], v14, v16, v13, v18, 0, 1);
    *a2 = v10;
    return v8;
  }
  if ( v5 != 507 )
  {
LABEL_21:
    v23[0] = "expected field label here";
    v24 = 259;
    goto LABEL_22;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "tag") )
    {
      v6 = sub_1208B00(a1, (__int64)"tag", 3, (__int64)v20);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "name") )
    {
      v6 = sub_120BB20(a1, "name", 4, (__int64)&v14);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "type") )
    {
      v6 = sub_1225DC0(a1, (__int64)"type", 4, (__int64)&v16);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "defaulted") )
      break;
    v6 = sub_1207D20(a1, (__int64)"defaulted", 9, (__int64)&v13);
LABEL_6:
    if ( v6 )
      return 1;
    if ( *(_DWORD *)(a1 + 240) != 4 )
      goto LABEL_8;
    v12 = sub_1205200(v4);
    *(_DWORD *)(a1 + 240) = v12;
    if ( v12 != 507 )
      goto LABEL_21;
  }
  if ( !(unsigned int)sub_2241AC0(a1 + 248, "value") )
  {
    v6 = sub_1225DC0(a1, (__int64)"value", 5, (__int64)&v18);
    goto LABEL_6;
  }
  v21[2] = a1 + 248;
  v21[0] = "invalid field '";
  v23[0] = v21;
  v22 = 1027;
  v23[2] = "'";
  v24 = 770;
LABEL_22:
  sub_11FD800(v4, *(_QWORD *)(a1 + 232), (__int64)v23, 1);
  return 1;
}

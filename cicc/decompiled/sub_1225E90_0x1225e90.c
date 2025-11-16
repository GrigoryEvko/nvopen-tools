// Function: sub_1225E90
// Address: 0x1225e90
//
__int64 __fastcall sub_1225E90(__int64 a1, _QWORD *a2, char a3)
{
  __int64 v4; // r13
  int v5; // eax
  char v6; // al
  unsigned __int64 v7; // r14
  unsigned int v8; // r15d
  __int64 *v9; // rdi
  _QWORD *v10; // rax
  int v12; // eax
  __int16 v13; // [rsp+1Eh] [rbp-F2h] BYREF
  __int64 v14; // [rsp+20h] [rbp-F0h] BYREF
  __int16 v15; // [rsp+28h] [rbp-E8h]
  __int64 v16; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v17; // [rsp+38h] [rbp-D8h]
  _QWORD v18[4]; // [rsp+40h] [rbp-D0h] BYREF
  _QWORD v19[4]; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD v20[4]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v21; // [rsp+A0h] [rbp-70h]
  _QWORD v22[4]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v23; // [rsp+D0h] [rbp-40h]

  v13 = 0;
  v4 = a1 + 176;
  v18[0] = 0;
  v18[1] = 0;
  v18[2] = 0xFFFFFFFFLL;
  v19[0] = 0;
  v19[1] = 0;
  v19[2] = 0xFFFF;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 256;
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
    if ( !(_BYTE)v15 )
    {
      v22[0] = "missing required field 'scope'";
      v23 = 259;
      sub_11FD800(v4, v7, (__int64)v22, 1);
      return 1;
    }
    v9 = *(__int64 **)a1;
    if ( a3 )
      v10 = sub_B01860(v9, v18[0], v19[0], v14, v16, v13, 1u, 1);
    else
      v10 = sub_B01860(v9, v18[0], v19[0], v14, v16, v13, 0, 1);
    *a2 = v10;
    return v8;
  }
  if ( v5 != 507 )
  {
LABEL_21:
    v22[0] = "expected field label here";
    v23 = 259;
    goto LABEL_22;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "line") )
    {
      v6 = sub_1208380(a1, (__int64)"line", 4, (__int64)v18);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "column") )
    {
      v6 = sub_12082C0(a1, (__int64)"column", 6, (__int64)v19);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "scope") )
    {
      v6 = sub_1225DC0(a1, (__int64)"scope", 5, (__int64)&v14);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "inlinedAt") )
      break;
    v6 = sub_1225DC0(a1, (__int64)"inlinedAt", 9, (__int64)&v16);
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
  if ( !(unsigned int)sub_2241AC0(a1 + 248, "isImplicitCode") )
  {
    v6 = sub_1207D20(a1, (__int64)"isImplicitCode", 14, (__int64)&v13);
    goto LABEL_6;
  }
  v20[2] = a1 + 248;
  v20[0] = "invalid field '";
  v22[0] = v20;
  v21 = 1027;
  v22[2] = "'";
  v23 = 770;
LABEL_22:
  sub_11FD800(v4, *(_QWORD *)(a1 + 232), (__int64)v22, 1);
  return 1;
}

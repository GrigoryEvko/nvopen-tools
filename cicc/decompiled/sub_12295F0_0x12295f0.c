// Function: sub_12295F0
// Address: 0x12295f0
//
__int64 __fastcall sub_12295F0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v5; // eax
  char v6; // al
  unsigned __int64 v7; // r14
  unsigned int v8; // r15d
  __int64 *v9; // rdi
  __int64 v10; // rax
  int v12; // eax
  __int16 v13; // [rsp+1Eh] [rbp-B2h] BYREF
  __int64 v14; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v15; // [rsp+28h] [rbp-A8h]
  __int64 v16; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v17; // [rsp+38h] [rbp-98h]
  _QWORD v18[4]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v19; // [rsp+60h] [rbp-70h]
  _QWORD v20[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v21; // [rsp+90h] [rbp-40h]

  v3 = a1 + 176;
  v17 = 256;
  v14 = 0;
  v15 = 256;
  v16 = 0;
  v13 = 0;
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
      v20[0] = "missing required field 'scope'";
      v21 = 259;
      sub_11FD800(v3, v7, (__int64)v20, 1);
      return 1;
    }
    v9 = *(__int64 **)a1;
    if ( a3 )
      v10 = sub_B096D0(v9, v14, v16, v13, 1u, 1);
    else
      v10 = sub_B096D0(v9, v14, v16, v13, 0, 1);
    *a2 = v10;
    return v8;
  }
  if ( v5 != 507 )
  {
LABEL_21:
    v20[0] = "expected field label here";
    v21 = 259;
    goto LABEL_24;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "scope") )
    {
      v6 = sub_1225DC0(a1, (__int64)"scope", 5, (__int64)&v14);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "name") )
      break;
    v6 = sub_120BB20(a1, "name", 4, (__int64)&v16);
LABEL_6:
    if ( v6 )
      return 1;
    if ( *(_DWORD *)(a1 + 240) != 4 )
      goto LABEL_8;
    v12 = sub_1205200(v3);
    *(_DWORD *)(a1 + 240) = v12;
    if ( v12 != 507 )
      goto LABEL_21;
  }
  if ( !(unsigned int)sub_2241AC0(a1 + 248, "exportSymbols") )
  {
    v6 = sub_1207D20(a1, (__int64)"exportSymbols", 13, (__int64)&v13);
    goto LABEL_6;
  }
  v18[2] = a1 + 248;
  v18[0] = "invalid field '";
  v19 = 1027;
  v20[0] = v18;
  v20[2] = "'";
  v21 = 770;
LABEL_24:
  sub_11FD800(v3, *(_QWORD *)(a1 + 232), (__int64)v20, 1);
  return 1;
}

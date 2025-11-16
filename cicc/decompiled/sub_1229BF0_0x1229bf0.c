// Function: sub_1229BF0
// Address: 0x1229bf0
//
__int64 __fastcall sub_1229BF0(__int64 a1, _QWORD *a2, char a3)
{
  __int64 v4; // r13
  int v5; // eax
  char v6; // al
  unsigned __int64 v7; // r15
  unsigned int v8; // r14d
  __int64 *v9; // rdi
  _BYTE *v10; // rax
  const char *v11; // rax
  int v13; // eax
  __int16 v14; // [rsp+1Eh] [rbp-112h] BYREF
  __int64 v15; // [rsp+20h] [rbp-110h] BYREF
  __int16 v16; // [rsp+28h] [rbp-108h]
  __int64 v17; // [rsp+30h] [rbp-100h] BYREF
  __int16 v18; // [rsp+38h] [rbp-F8h]
  __int64 v19; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v20; // [rsp+48h] [rbp-E8h]
  __int64 v21; // [rsp+50h] [rbp-E0h] BYREF
  __int16 v22; // [rsp+58h] [rbp-D8h]
  __int64 v23; // [rsp+60h] [rbp-D0h] BYREF
  __int16 v24; // [rsp+68h] [rbp-C8h]
  __int64 v25; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v26; // [rsp+78h] [rbp-B8h]
  _QWORD v27[4]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD v28[4]; // [rsp+A0h] [rbp-90h] BYREF
  __int16 v29; // [rsp+C0h] [rbp-70h]
  _QWORD v30[4]; // [rsp+D0h] [rbp-60h] BYREF
  __int16 v31; // [rsp+F0h] [rbp-40h]

  v14 = 0;
  v4 = a1 + 176;
  v18 = 256;
  v15 = 0;
  v16 = 256;
  v17 = 0;
  v19 = 0;
  v20 = 256;
  v21 = 0;
  v22 = 256;
  v23 = 0;
  v24 = 256;
  v25 = 0;
  v26 = 256;
  v27[0] = 0;
  v27[1] = 0;
  v27[2] = 0xFFFFFFFFLL;
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
    if ( (_BYTE)v16 )
    {
      if ( (_BYTE)v18 )
      {
        v9 = *(__int64 **)a1;
        if ( a3 )
          v10 = sub_B0A3E0(v9, v25, v15, v17, v19, v21, v23, v27[0], v14, 1u, 1);
        else
          v10 = sub_B0A3E0(v9, v25, v15, v17, v19, v21, v23, v27[0], v14, 0, 1);
        *a2 = v10;
        return v8;
      }
      HIBYTE(v31) = 1;
      v11 = "missing required field 'name'";
    }
    else
    {
      HIBYTE(v31) = 1;
      v11 = "missing required field 'scope'";
    }
    v30[0] = v11;
    LOBYTE(v31) = 3;
    sub_11FD800(v4, v7, (__int64)v30, 1);
    return 1;
  }
  if ( v5 != 507 )
  {
LABEL_22:
    v30[0] = "expected field label here";
    v31 = 259;
    goto LABEL_23;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "scope") )
    {
      v6 = sub_1225DC0(a1, (__int64)"scope", 5, (__int64)&v15);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "name") )
    {
      v6 = sub_120BB20(a1, "name", 4, (__int64)&v17);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "configMacros") )
    {
      v6 = sub_120BB20(a1, "configMacros", 12, (__int64)&v19);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "includePath") )
    {
      v6 = sub_120BB20(a1, "includePath", 11, (__int64)&v21);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "apinotes") )
    {
      v6 = sub_120BB20(a1, "apinotes", 8, (__int64)&v23);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "file") )
    {
      v6 = sub_1225DC0(a1, (__int64)"file", 4, (__int64)&v25);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "line") )
    {
      v6 = sub_1208380(a1, (__int64)"line", 4, (__int64)v27);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "isDecl") )
      break;
    v6 = sub_1207D20(a1, (__int64)"isDecl", 6, (__int64)&v14);
LABEL_6:
    if ( v6 )
      return 1;
    if ( *(_DWORD *)(a1 + 240) != 4 )
      goto LABEL_8;
    v13 = sub_1205200(v4);
    *(_DWORD *)(a1 + 240) = v13;
    if ( v13 != 507 )
      goto LABEL_22;
  }
  v28[2] = a1 + 248;
  v28[0] = "invalid field '";
  v29 = 1027;
  v30[0] = v28;
  v30[2] = "'";
  v31 = 770;
LABEL_23:
  sub_11FD800(v4, *(_QWORD *)(a1 + 232), (__int64)v30, 1);
  return 1;
}

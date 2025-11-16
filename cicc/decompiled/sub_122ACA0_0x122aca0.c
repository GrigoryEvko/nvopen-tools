// Function: sub_122ACA0
// Address: 0x122aca0
//
__int64 __fastcall sub_122ACA0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v5; // eax
  char v6; // al
  unsigned __int64 v7; // r14
  unsigned int v8; // r15d
  _QWORD *v9; // rdi
  __int64 v10; // rax
  int v12; // eax
  int v13; // [rsp+18h] [rbp-148h] BYREF
  char v14; // [rsp+1Ch] [rbp-144h]
  __int64 v15; // [rsp+20h] [rbp-140h] BYREF
  __int16 v16; // [rsp+28h] [rbp-138h]
  __int64 v17; // [rsp+30h] [rbp-130h] BYREF
  __int16 v18; // [rsp+38h] [rbp-128h]
  __int64 v19; // [rsp+40h] [rbp-120h] BYREF
  __int16 v20; // [rsp+48h] [rbp-118h]
  __int64 v21; // [rsp+50h] [rbp-110h] BYREF
  __int16 v22; // [rsp+58h] [rbp-108h]
  __int64 v23; // [rsp+60h] [rbp-100h] BYREF
  __int16 v24; // [rsp+68h] [rbp-F8h]
  _QWORD v25[4]; // [rsp+70h] [rbp-F0h] BYREF
  _QWORD v26[4]; // [rsp+90h] [rbp-D0h] BYREF
  _QWORD v27[4]; // [rsp+B0h] [rbp-B0h] BYREF
  _QWORD v28[4]; // [rsp+D0h] [rbp-90h] BYREF
  __int16 v29; // [rsp+F0h] [rbp-70h]
  _QWORD v30[4]; // [rsp+100h] [rbp-60h] BYREF
  __int16 v31; // [rsp+120h] [rbp-40h]

  v3 = a1 + 176;
  v18 = 256;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v25[0] = 0;
  v25[1] = 0;
  v25[2] = 0xFFFF;
  v19 = 0;
  v20 = 256;
  v26[0] = 0;
  v26[1] = 0;
  v26[2] = 0xFFFFFFFFLL;
  v21 = 0;
  v22 = 256;
  v13 = 0;
  v14 = 0;
  v27[0] = 0;
  v27[1] = 0;
  v27[2] = 0xFFFFFFFFLL;
  v23 = 0;
  v24 = 256;
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
    if ( !(_BYTE)v16 )
    {
      v30[0] = "missing required field 'scope'";
      v31 = 259;
      sub_11FD800(v3, v7, (__int64)v30, 1);
      return 1;
    }
    v9 = *(_QWORD **)a1;
    if ( a3 )
      v10 = sub_B0C150(v9, v15, v17, v19, v26[0], v21, v25[0], v13, v27[0], v23, 1u, 1);
    else
      v10 = sub_B0C150(v9, v15, v17, v19, v26[0], v21, v25[0], v13, v27[0], v23, 0, 1);
    *a2 = v10;
    return v8;
  }
  if ( v5 != 507 )
  {
LABEL_21:
    v30[0] = "expected field label here";
    v31 = 259;
    goto LABEL_22;
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
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "arg") )
    {
      v6 = sub_1208450(a1, (__int64)"arg", 3, (__int64)v25);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "file") )
    {
      v6 = sub_1225DC0(a1, (__int64)"file", 4, (__int64)&v19);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "line") )
    {
      v6 = sub_1208380(a1, (__int64)"line", 4, (__int64)v26);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "type") )
    {
      v6 = sub_1225DC0(a1, (__int64)"type", 4, (__int64)&v21);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "flags") )
    {
      v6 = sub_120BE50(a1, (__int64)"flags", 5, (__int64)&v13);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "align") )
    {
      v6 = sub_1208450(a1, (__int64)"align", 5, (__int64)v27);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "annotations") )
      break;
    v6 = sub_1225DC0(a1, (__int64)"annotations", 11, (__int64)&v23);
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
  v28[2] = a1 + 248;
  v28[0] = "invalid field '";
  v29 = 1027;
  v30[0] = v28;
  v30[2] = "'";
  v31 = 770;
LABEL_22:
  sub_11FD800(v3, *(_QWORD *)(a1 + 232), (__int64)v30, 1);
  return 1;
}

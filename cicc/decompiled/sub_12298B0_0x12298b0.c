// Function: sub_12298B0
// Address: 0x12298b0
//
__int64 __fastcall sub_12298B0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v5; // eax
  char v6; // al
  unsigned __int64 v7; // r14
  unsigned int v8; // r15d
  __int64 *v9; // rdi
  __int64 v10; // rax
  int v12; // eax
  __int64 v13; // [rsp+10h] [rbp-F0h] BYREF
  __int16 v14; // [rsp+18h] [rbp-E8h]
  __int64 v15; // [rsp+20h] [rbp-E0h] BYREF
  __int16 v16; // [rsp+28h] [rbp-D8h]
  _QWORD v17[4]; // [rsp+30h] [rbp-D0h] BYREF
  _QWORD v18[4]; // [rsp+50h] [rbp-B0h] BYREF
  _QWORD v19[4]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v20; // [rsp+90h] [rbp-70h]
  _QWORD v21[4]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v22; // [rsp+C0h] [rbp-40h]

  v3 = a1 + 176;
  v17[0] = 3;
  v17[1] = 0;
  v17[2] = 255;
  v18[0] = 0;
  v18[1] = 0;
  v18[2] = 0xFFFFFFFFLL;
  v13 = 0;
  v14 = 256;
  v15 = 0;
  v16 = 256;
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
    if ( !(_BYTE)v14 )
    {
      v21[0] = "missing required field 'file'";
      v22 = 259;
      sub_11FD800(v3, v7, (__int64)v21, 1);
      return 1;
    }
    v9 = *(__int64 **)a1;
    if ( a3 )
      v10 = sub_B10A70(v9, v17[0], v18[0], v13, v15, 1u, 1);
    else
      v10 = sub_B10A70(v9, v17[0], v18[0], v13, v15, 0, 1);
    *a2 = v10;
    return v8;
  }
  if ( v5 != 507 )
  {
LABEL_21:
    v21[0] = "expected field label here";
    v22 = 259;
    goto LABEL_25;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "type") )
    {
      v6 = sub_1208920(a1, (__int64)"type", 4, (__int64)v17);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "line") )
    {
      v6 = sub_1208380(a1, (__int64)"line", 4, (__int64)v18);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "file") )
    {
      v6 = sub_1225DC0(a1, (__int64)"file", 4, (__int64)&v13);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "nodes") )
      break;
    v6 = sub_1225DC0(a1, (__int64)"nodes", 5, (__int64)&v15);
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
  v19[2] = a1 + 248;
  v19[0] = "invalid field '";
  v21[0] = v19;
  v20 = 1027;
  v21[2] = "'";
  v22 = 770;
LABEL_25:
  sub_11FD800(v3, *(_QWORD *)(a1 + 232), (__int64)v21, 1);
  return 1;
}

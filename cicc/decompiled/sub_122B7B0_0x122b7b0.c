// Function: sub_122B7B0
// Address: 0x122b7b0
//
__int64 __fastcall sub_122B7B0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v5; // eax
  char v6; // al
  unsigned int v7; // r13d
  __int64 *v8; // rdi
  __int64 v9; // rax
  int v11; // eax
  __int64 v12; // [rsp+10h] [rbp-120h] BYREF
  __int16 v13; // [rsp+18h] [rbp-118h]
  __int64 v14; // [rsp+20h] [rbp-110h] BYREF
  __int16 v15; // [rsp+28h] [rbp-108h]
  __int64 v16; // [rsp+30h] [rbp-100h] BYREF
  __int16 v17; // [rsp+38h] [rbp-F8h]
  __int64 v18; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v19; // [rsp+48h] [rbp-E8h]
  __int64 v20; // [rsp+50h] [rbp-E0h] BYREF
  __int16 v21; // [rsp+58h] [rbp-D8h]
  _QWORD v22[4]; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD v23[4]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD v24[4]; // [rsp+A0h] [rbp-90h] BYREF
  __int16 v25; // [rsp+C0h] [rbp-70h]
  _QWORD v26[4]; // [rsp+D0h] [rbp-60h] BYREF
  __int16 v27; // [rsp+F0h] [rbp-40h]

  v3 = a1 + 176;
  v15 = 256;
  v12 = 0;
  v13 = 256;
  v14 = 0;
  v22[0] = 0;
  v22[1] = 0;
  v22[2] = 0xFFFFFFFFLL;
  v16 = 0;
  v17 = 256;
  v18 = 0;
  v19 = 256;
  v23[0] = 0;
  v23[1] = 0;
  v23[2] = 0xFFFFFFFFLL;
  v20 = 0;
  v21 = 256;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    return 1;
  v5 = *(_DWORD *)(a1 + 240);
  if ( v5 == 13 )
  {
LABEL_8:
    v7 = sub_120AFE0(a1, 13, "expected ')' here");
    if ( (_BYTE)v7 )
      return 1;
    v8 = *(__int64 **)a1;
    if ( a3 )
      v9 = sub_B0F520(v8, v12, v14, v22[0], v16, v18, v23[0], v20, 1u, 1);
    else
      v9 = sub_B0F520(v8, v12, v14, v22[0], v16, v18, v23[0], v20, 0, 1);
    *a2 = v9;
    return v7;
  }
  if ( v5 != 507 )
  {
LABEL_15:
    v26[0] = "expected field label here";
    v27 = 259;
    goto LABEL_16;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "name") )
    {
      v6 = sub_120BB20(a1, "name", 4, (__int64)&v12);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "file") )
    {
      v6 = sub_1225DC0(a1, (__int64)"file", 4, (__int64)&v14);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "line") )
    {
      v6 = sub_1208380(a1, (__int64)"line", 4, (__int64)v22);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "setter") )
    {
      v6 = sub_120BB20(a1, "setter", 6, (__int64)&v16);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "getter") )
    {
      v6 = sub_120BB20(a1, "getter", 6, (__int64)&v18);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "attributes") )
    {
      v6 = sub_1208450(a1, (__int64)"attributes", 10, (__int64)v23);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "type") )
      break;
    v6 = sub_1225DC0(a1, (__int64)"type", 4, (__int64)&v20);
LABEL_6:
    if ( v6 )
      return 1;
    if ( *(_DWORD *)(a1 + 240) != 4 )
      goto LABEL_8;
    v11 = sub_1205200(v3);
    *(_DWORD *)(a1 + 240) = v11;
    if ( v11 != 507 )
      goto LABEL_15;
  }
  v24[2] = a1 + 248;
  v24[0] = "invalid field '";
  v25 = 1027;
  v26[0] = v24;
  v26[2] = "'";
  v27 = 770;
LABEL_16:
  sub_11FD800(v3, *(_QWORD *)(a1 + 232), (__int64)v26, 1);
  return 1;
}

// Function: sub_120F380
// Address: 0x120f380
//
__int64 __fastcall sub_120F380(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v5; // eax
  char v6; // al
  unsigned int v7; // r13d
  __int64 *v8; // rdi
  __int64 v9; // rax
  int v11; // eax
  int v12; // [rsp+18h] [rbp-148h] BYREF
  char v13; // [rsp+1Ch] [rbp-144h]
  __int64 v14; // [rsp+20h] [rbp-140h] BYREF
  __int16 v15; // [rsp+28h] [rbp-138h]
  _QWORD v16[4]; // [rsp+30h] [rbp-130h] BYREF
  __int64 v17[4]; // [rsp+50h] [rbp-110h] BYREF
  _QWORD v18[4]; // [rsp+70h] [rbp-F0h] BYREF
  _QWORD v19[4]; // [rsp+90h] [rbp-D0h] BYREF
  _QWORD v20[4]; // [rsp+B0h] [rbp-B0h] BYREF
  _QWORD v21[4]; // [rsp+D0h] [rbp-90h] BYREF
  __int16 v22; // [rsp+F0h] [rbp-70h]
  _QWORD v23[4]; // [rsp+100h] [rbp-60h] BYREF
  __int16 v24; // [rsp+120h] [rbp-40h]

  v3 = a1 + 176;
  v15 = 256;
  v16[0] = 36;
  v16[1] = 0;
  v16[2] = 0xFFFF;
  v14 = 0;
  v17[0] = 0;
  v17[1] = 0;
  v17[2] = -1;
  v18[0] = 0;
  v18[1] = 0;
  v18[2] = 0xFFFFFFFFLL;
  v19[0] = 0;
  v19[1] = 0;
  v19[2] = 255;
  v20[0] = 0;
  v20[1] = 0;
  v20[2] = 0xFFFFFFFFLL;
  v12 = 0;
  v13 = 0;
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
      v9 = sub_B04E90(v8, v16[0], v14, v17[0], v18[0], v19[0], v20[0], v12, 1u, 1);
    else
      v9 = sub_B04E90(v8, v16[0], v14, v17[0], v18[0], v19[0], v20[0], v12, 0, 1);
    *a2 = v9;
    return v7;
  }
  if ( v5 != 507 )
  {
LABEL_15:
    v23[0] = "expected field label here";
    v24 = 259;
    goto LABEL_16;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "tag") )
    {
      v6 = sub_1208B00(a1, (__int64)"tag", 3, (__int64)v16);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "name") )
    {
      v6 = sub_120BB20(a1, "name", 4, (__int64)&v14);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "size") )
    {
      v6 = sub_1208450(a1, (__int64)"size", 4, (__int64)v17);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "align") )
    {
      v6 = sub_1208450(a1, (__int64)"align", 5, (__int64)v18);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "encoding") )
    {
      v6 = sub_1208520(a1, (__int64)"encoding", 8, (__int64)v19);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "num_extra_inhabitants") )
    {
      v6 = sub_1208450(a1, (__int64)"num_extra_inhabitants", 21, (__int64)v20);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "flags") )
      break;
    v6 = sub_120BE50(a1, (__int64)"flags", 5, (__int64)&v12);
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
  v21[2] = a1 + 248;
  v21[0] = "invalid field '";
  v22 = 1027;
  v23[0] = v21;
  v23[2] = "'";
  v24 = 770;
LABEL_16:
  sub_11FD800(v3, *(_QWORD *)(a1 + 232), (__int64)v23, 1);
  return 1;
}

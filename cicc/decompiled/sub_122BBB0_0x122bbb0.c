// Function: sub_122BBB0
// Address: 0x122bbb0
//
__int64 __fastcall sub_122BBB0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v5; // eax
  char v6; // al
  unsigned __int64 v7; // r15
  unsigned int v8; // r14d
  _QWORD *v9; // rdi
  __int64 v10; // rax
  const char *v11; // rax
  int v13; // eax
  __int64 v14; // [rsp+10h] [rbp-120h] BYREF
  __int16 v15; // [rsp+18h] [rbp-118h]
  __int64 v16; // [rsp+20h] [rbp-110h] BYREF
  __int16 v17; // [rsp+28h] [rbp-108h]
  __int64 v18; // [rsp+30h] [rbp-100h] BYREF
  __int16 v19; // [rsp+38h] [rbp-F8h]
  __int64 v20; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v21; // [rsp+48h] [rbp-E8h]
  __int64 v22; // [rsp+50h] [rbp-E0h] BYREF
  __int16 v23; // [rsp+58h] [rbp-D8h]
  __int64 v24; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v25; // [rsp+68h] [rbp-C8h]
  __int64 v26; // [rsp+70h] [rbp-C0h]
  _QWORD v27[4]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD v28[4]; // [rsp+A0h] [rbp-90h] BYREF
  __int16 v29; // [rsp+C0h] [rbp-70h]
  _QWORD v30[4]; // [rsp+D0h] [rbp-60h] BYREF
  __int16 v31; // [rsp+F0h] [rbp-40h]

  v3 = a1 + 176;
  v17 = 256;
  v24 = 0;
  v25 = 0;
  v26 = 0xFFFF;
  v14 = 0;
  v15 = 256;
  v16 = 0;
  v18 = 0;
  v19 = 256;
  v27[0] = 0;
  v27[1] = 0;
  v27[2] = 0xFFFFFFFFLL;
  v20 = 0;
  v21 = 256;
  v22 = 0;
  v23 = 256;
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
    if ( (_BYTE)v25 )
    {
      if ( (_BYTE)v15 )
      {
        v9 = *(_QWORD **)a1;
        if ( a3 )
          v10 = sub_B0FC60(v9, v24, v14, v16, v18, v27[0], v20, v22, 1u, 1);
        else
          v10 = sub_B0FC60(v9, v24, v14, v16, v18, v27[0], v20, v22, 0, 1);
        *a2 = v10;
        return v8;
      }
      HIBYTE(v31) = 1;
      v11 = "missing required field 'scope'";
    }
    else
    {
      HIBYTE(v31) = 1;
      v11 = "missing required field 'tag'";
    }
    v30[0] = v11;
    LOBYTE(v31) = 3;
    sub_11FD800(v3, v7, (__int64)v30, 1);
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
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "tag") )
    {
      v6 = sub_1208B00(a1, (__int64)"tag", 3, (__int64)&v24);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "scope") )
    {
      v6 = sub_1225DC0(a1, (__int64)"scope", 5, (__int64)&v14);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "entity") )
    {
      v6 = sub_1225DC0(a1, (__int64)"entity", 6, (__int64)&v16);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "file") )
    {
      v6 = sub_1225DC0(a1, (__int64)"file", 4, (__int64)&v18);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "line") )
    {
      v6 = sub_1208380(a1, (__int64)"line", 4, (__int64)v27);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "name") )
    {
      v6 = sub_120BB20(a1, "name", 4, (__int64)&v20);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "elements") )
      break;
    v6 = sub_1225DC0(a1, (__int64)"elements", 8, (__int64)&v22);
LABEL_6:
    if ( v6 )
      return 1;
    if ( *(_DWORD *)(a1 + 240) != 4 )
      goto LABEL_8;
    v13 = sub_1205200(v3);
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
  sub_11FD800(v3, *(_QWORD *)(a1 + 232), (__int64)v30, 1);
  return 1;
}

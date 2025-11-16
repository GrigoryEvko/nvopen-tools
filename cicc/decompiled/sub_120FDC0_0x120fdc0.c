// Function: sub_120FDC0
// Address: 0x120fdc0
//
__int64 __fastcall sub_120FDC0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v5; // eax
  char v6; // al
  unsigned __int64 v7; // r15
  unsigned int v8; // r14d
  __int64 *v9; // rdi
  __int64 v10; // rax
  const char *v11; // rax
  int v13; // eax
  __int64 v14; // [rsp+10h] [rbp-F0h] BYREF
  __int16 v15; // [rsp+18h] [rbp-E8h]
  __int64 v16; // [rsp+20h] [rbp-E0h] BYREF
  __int16 v17; // [rsp+28h] [rbp-D8h]
  __int64 v18; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v19; // [rsp+38h] [rbp-C8h]
  __int64 v20; // [rsp+40h] [rbp-C0h]
  _QWORD v21[4]; // [rsp+50h] [rbp-B0h] BYREF
  _QWORD v22[4]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v23; // [rsp+90h] [rbp-70h]
  _QWORD v24[4]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v25; // [rsp+C0h] [rbp-40h]

  v3 = a1 + 176;
  v18 = 0;
  v19 = 0;
  v20 = 255;
  v21[0] = 0;
  v21[1] = 0;
  v21[2] = 0xFFFFFFFFLL;
  v14 = 0;
  v15 = 256;
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
    if ( (_BYTE)v19 )
    {
      if ( (_BYTE)v15 )
      {
        v9 = *(__int64 **)a1;
        if ( a3 )
          v10 = sub_B103D0(v9, v18, v21[0], v14, v16, 1u, 1);
        else
          v10 = sub_B103D0(v9, v18, v21[0], v14, v16, 0, 1);
        *a2 = v10;
        return v8;
      }
      HIBYTE(v25) = 1;
      v11 = "missing required field 'name'";
    }
    else
    {
      HIBYTE(v25) = 1;
      v11 = "missing required field 'type'";
    }
    v24[0] = v11;
    LOBYTE(v25) = 3;
    sub_11FD800(v3, v7, (__int64)v24, 1);
    return 1;
  }
  if ( v5 != 507 )
  {
LABEL_22:
    v24[0] = "expected field label here";
    v25 = 259;
    goto LABEL_28;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "type") )
    {
      v6 = sub_1208920(a1, (__int64)"type", 4, (__int64)&v18);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "line") )
    {
      v6 = sub_1208380(a1, (__int64)"line", 4, (__int64)v21);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "name") )
    {
      v6 = sub_120BB20(a1, "name", 4, (__int64)&v14);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "value") )
      break;
    v6 = sub_120BB20(a1, "value", 5, (__int64)&v16);
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
  v22[2] = a1 + 248;
  v22[0] = "invalid field '";
  v24[0] = v22;
  v23 = 1027;
  v24[2] = "'";
  v25 = 770;
LABEL_28:
  sub_11FD800(v3, *(_QWORD *)(a1 + 232), (__int64)v24, 1);
  return 1;
}

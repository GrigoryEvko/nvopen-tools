// Function: sub_1228F60
// Address: 0x1228f60
//
__int64 __fastcall sub_1228F60(__int64 a1, __int64 *a2, char a3)
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
  __int64 v14; // [rsp+10h] [rbp-D0h] BYREF
  __int16 v15; // [rsp+18h] [rbp-C8h]
  __int64 v16; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v17; // [rsp+28h] [rbp-B8h]
  __int64 v18; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v19; // [rsp+38h] [rbp-A8h]
  __int64 v20; // [rsp+40h] [rbp-A0h]
  _QWORD v21[4]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v22; // [rsp+70h] [rbp-70h]
  _QWORD v23[4]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v24; // [rsp+A0h] [rbp-40h]

  v3 = a1 + 176;
  v17 = 256;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0xFFFFFFFFLL;
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
    if ( (_BYTE)v15 )
    {
      if ( (_BYTE)v19 )
      {
        v9 = *(__int64 **)a1;
        if ( a3 )
          v10 = sub_B09060(v9, v14, v16, v18, 1u, 1);
        else
          v10 = sub_B09060(v9, v14, v16, v18, 0, 1);
        *a2 = v10;
        return v8;
      }
      HIBYTE(v24) = 1;
      v11 = "missing required field 'discriminator'";
    }
    else
    {
      HIBYTE(v24) = 1;
      v11 = "missing required field 'scope'";
    }
    v23[0] = v11;
    LOBYTE(v24) = 3;
    sub_11FD800(v3, v7, (__int64)v23, 1);
    return 1;
  }
  if ( v5 != 507 )
  {
LABEL_22:
    v23[0] = "expected field label here";
    v24 = 259;
    goto LABEL_27;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "scope") )
    {
      v6 = sub_1225DC0(a1, (__int64)"scope", 5, (__int64)&v14);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "file") )
      break;
    v6 = sub_1225DC0(a1, (__int64)"file", 4, (__int64)&v16);
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
  if ( !(unsigned int)sub_2241AC0(a1 + 248, "discriminator") )
  {
    v6 = sub_1208450(a1, (__int64)"discriminator", 13, (__int64)&v18);
    goto LABEL_6;
  }
  v21[2] = a1 + 248;
  v21[0] = "invalid field '";
  v22 = 1027;
  v23[0] = v21;
  v23[2] = "'";
  v24 = 770;
LABEL_27:
  sub_11FD800(v3, *(_QWORD *)(a1 + 232), (__int64)v23, 1);
  return 1;
}

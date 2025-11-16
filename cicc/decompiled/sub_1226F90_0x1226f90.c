// Function: sub_1226F90
// Address: 0x1226f90
//
__int64 __fastcall sub_1226F90(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v5; // eax
  char v6; // al
  int v7; // eax
  unsigned __int64 v8; // r14
  unsigned int v9; // r15d
  __int64 *v10; // rdi
  __int64 v11; // rax
  unsigned __int64 v13; // rsi
  int v14; // eax
  int v15; // eax
  unsigned int v16; // eax
  int v17; // [rsp+18h] [rbp-F8h] BYREF
  char v18; // [rsp+1Ch] [rbp-F4h]
  __int64 v19; // [rsp+20h] [rbp-F0h] BYREF
  __int16 v20; // [rsp+28h] [rbp-E8h]
  __int64 v21; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v22; // [rsp+38h] [rbp-D8h]
  __int64 v23; // [rsp+40h] [rbp-D0h]
  _QWORD v24[4]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v25; // [rsp+70h] [rbp-A0h]
  _QWORD v26[2]; // [rsp+80h] [rbp-90h] BYREF
  const char *v27; // [rsp+90h] [rbp-80h]
  __int16 v28; // [rsp+A0h] [rbp-70h]
  _QWORD v29[2]; // [rsp+B0h] [rbp-60h] BYREF
  char *v30; // [rsp+C0h] [rbp-50h]
  __int64 v31; // [rsp+C8h] [rbp-48h]
  __int16 v32; // [rsp+D0h] [rbp-40h]

  v3 = a1 + 176;
  v17 = 0;
  v18 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 255;
  v19 = 0;
  v20 = 256;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    return 1;
  v5 = *(_DWORD *)(a1 + 240);
  if ( v5 == 13 )
  {
LABEL_9:
    v8 = *(_QWORD *)(a1 + 232);
    v9 = sub_120AFE0(a1, 13, "expected ')' here");
    if ( (_BYTE)v9 )
      return 1;
    if ( !(_BYTE)v20 )
    {
      v29[0] = "missing required field 'types'";
      v32 = 259;
      sub_11FD800(v3, v8, (__int64)v29, 1);
      return 1;
    }
    v10 = *(__int64 **)a1;
    if ( a3 )
      v11 = sub_B07260(v10, v17, v21, v19, 1u, 1);
    else
      v11 = sub_B07260(v10, v17, v21, v19, 0, 1);
    *a2 = v11;
    return v9;
  }
  if ( v5 != 507 )
  {
LABEL_22:
    v29[0] = "expected field label here";
    v32 = 259;
    goto LABEL_27;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "flags") )
    {
      v6 = sub_120BE50(a1, (__int64)"flags", 5, (__int64)&v17);
LABEL_6:
      if ( v6 )
        return 1;
      v7 = *(_DWORD *)(a1 + 240);
      goto LABEL_8;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "cc") )
    {
      if ( (unsigned int)sub_2241AC0(a1 + 248, "types") )
      {
        v27 = (const char *)(a1 + 248);
        v26[0] = "invalid field '";
        v28 = 1027;
LABEL_26:
        v29[0] = v26;
        v30 = "'";
        v32 = 770;
        goto LABEL_27;
      }
      v6 = sub_1225DC0(a1, (__int64)"types", 5, (__int64)&v19);
      goto LABEL_6;
    }
    if ( (_BYTE)v22 )
    {
      v13 = *(_QWORD *)(a1 + 232);
      v29[0] = "field '";
      v30 = "cc";
      v32 = 1283;
      v26[0] = v29;
      v31 = 2;
      v27 = "' cannot be specified more than once";
      v28 = 770;
      sub_11FD800(v3, v13, (__int64)v26, 1);
      return 1;
    }
    v15 = sub_1205200(v3);
    *(_DWORD *)(a1 + 240) = v15;
    if ( v15 == 529 )
    {
      v6 = sub_1208110(a1, (__int64)"cc", 2, (__int64)&v21);
      goto LABEL_6;
    }
    if ( v15 != 517 )
      break;
    v16 = sub_E0BC20(*(_QWORD *)(a1 + 248), *(_QWORD *)(a1 + 256));
    if ( !v16 )
    {
      v27 = (const char *)(a1 + 248);
      v24[0] = "invalid DWARF calling convention";
      v24[2] = " '";
      v25 = 771;
      v26[0] = v24;
      v28 = 1026;
      goto LABEL_26;
    }
    LOBYTE(v22) = 1;
    v21 = v16;
    v7 = sub_1205200(v3);
    *(_DWORD *)(a1 + 240) = v7;
LABEL_8:
    if ( v7 != 4 )
      goto LABEL_9;
    v14 = sub_1205200(v3);
    *(_DWORD *)(a1 + 240) = v14;
    if ( v14 != 507 )
      goto LABEL_22;
  }
  v29[0] = "expected DWARF calling convention";
  v32 = 259;
LABEL_27:
  sub_11FD800(v3, *(_QWORD *)(a1 + 232), (__int64)v29, 1);
  return 1;
}

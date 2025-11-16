// Function: sub_122B170
// Address: 0x122b170
//
__int64 __fastcall sub_122B170(__int64 a1, __int64 *a2, char a3)
{
  __int64 v4; // r13
  int v5; // eax
  char v6; // al
  unsigned __int64 v7; // r15
  unsigned int v8; // r14d
  __int64 *v9; // rdi
  __int64 v10; // rax
  const char *v11; // rax
  int v13; // eax
  __int64 v14; // [rsp+10h] [rbp-E0h] BYREF
  __int16 v15; // [rsp+18h] [rbp-D8h]
  __int64 v16; // [rsp+20h] [rbp-D0h] BYREF
  __int16 v17; // [rsp+28h] [rbp-C8h]
  __int64 v18; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v19; // [rsp+38h] [rbp-B8h]
  __int64 v20; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v21; // [rsp+48h] [rbp-A8h]
  __int64 v22; // [rsp+50h] [rbp-A0h]
  _QWORD v23[4]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v24; // [rsp+80h] [rbp-70h]
  _QWORD v25[4]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v26; // [rsp+B0h] [rbp-40h]

  v19 = 256;
  v4 = a1 + 176;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 256;
  v18 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0xFFFFFFFFLL;
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
      if ( (_BYTE)v17 )
      {
        if ( (_BYTE)v19 )
        {
          if ( (_BYTE)v21 )
          {
            v9 = *(__int64 **)a1;
            if ( a3 )
              v10 = sub_B0C9F0(v9, v14, v16, v18, v20, 1u, 1);
            else
              v10 = sub_B0C9F0(v9, v14, v16, v18, v20, 0, 1);
            *a2 = v10;
            return v8;
          }
          HIBYTE(v26) = 1;
          v11 = "missing required field 'line'";
        }
        else
        {
          HIBYTE(v26) = 1;
          v11 = "missing required field 'file'";
        }
      }
      else
      {
        HIBYTE(v26) = 1;
        v11 = "missing required field 'name'";
      }
    }
    else
    {
      HIBYTE(v26) = 1;
      v11 = "missing required field 'scope'";
    }
    v25[0] = v11;
    LOBYTE(v26) = 3;
    sub_11FD800(v4, v7, (__int64)v25, 1);
    return 1;
  }
  if ( v5 != 507 )
  {
LABEL_24:
    v25[0] = "expected field label here";
    v26 = 259;
    goto LABEL_30;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "scope") )
    {
      v6 = sub_1225DC0(a1, (__int64)"scope", 5, (__int64)&v14);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "name") )
    {
      v6 = sub_120BB20(a1, "name", 4, (__int64)&v16);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "file") )
    {
      v6 = sub_1225DC0(a1, (__int64)"file", 4, (__int64)&v18);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "line") )
      break;
    v6 = sub_1208380(a1, (__int64)"line", 4, (__int64)&v20);
LABEL_6:
    if ( v6 )
      return 1;
    if ( *(_DWORD *)(a1 + 240) != 4 )
      goto LABEL_8;
    v13 = sub_1205200(v4);
    *(_DWORD *)(a1 + 240) = v13;
    if ( v13 != 507 )
      goto LABEL_24;
  }
  v23[2] = a1 + 248;
  v23[0] = "invalid field '";
  v25[0] = v23;
  v24 = 1027;
  v25[2] = "'";
  v26 = 770;
LABEL_30:
  sub_11FD800(v4, *(_QWORD *)(a1 + 232), (__int64)v25, 1);
  return 1;
}

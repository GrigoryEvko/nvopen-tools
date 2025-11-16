// Function: sub_122B510
// Address: 0x122b510
//
__int64 __fastcall sub_122B510(__int64 a1, __int64 *a2, char a3)
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
  __int64 v14; // [rsp+10h] [rbp-B0h] BYREF
  __int16 v15; // [rsp+18h] [rbp-A8h]
  __int64 v16; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v17; // [rsp+28h] [rbp-98h]
  _QWORD v18[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v19; // [rsp+50h] [rbp-70h]
  _QWORD v20[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v21; // [rsp+80h] [rbp-40h]

  v3 = a1 + 176;
  v17 = 256;
  v14 = 0;
  v15 = 256;
  v16 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    return 1;
  v5 = *(_DWORD *)(a1 + 240);
  if ( v5 != 13 )
  {
    if ( v5 == 507 )
    {
      do
      {
        if ( (unsigned int)sub_2241AC0(a1 + 248, "var") )
        {
          if ( (unsigned int)sub_2241AC0(a1 + 248, "expr") )
          {
            v18[2] = a1 + 248;
            v18[0] = "invalid field '";
            v19 = 1027;
            v20[0] = v18;
            v20[2] = "'";
            v21 = 770;
            goto LABEL_20;
          }
          v6 = sub_1225DC0(a1, (__int64)"expr", 4, (__int64)&v16);
        }
        else
        {
          v6 = sub_1225DC0(a1, (__int64)"var", 3, (__int64)&v14);
        }
        if ( v6 )
          return 1;
        if ( *(_DWORD *)(a1 + 240) != 4 )
          goto LABEL_8;
        v13 = sub_1205200(v3);
        *(_DWORD *)(a1 + 240) = v13;
      }
      while ( v13 == 507 );
    }
    v20[0] = "expected field label here";
    v21 = 259;
LABEL_20:
    sub_11FD800(v3, *(_QWORD *)(a1 + 232), (__int64)v20, 1);
    return 1;
  }
LABEL_8:
  v7 = *(_QWORD *)(a1 + 232);
  v8 = sub_120AFE0(a1, 13, "expected ')' here");
  if ( (_BYTE)v8 )
    return 1;
  if ( !(_BYTE)v15 )
  {
    HIBYTE(v21) = 1;
    v11 = "missing required field 'var'";
LABEL_15:
    v20[0] = v11;
    LOBYTE(v21) = 3;
    sub_11FD800(v3, v7, (__int64)v20, 1);
    return 1;
  }
  if ( !(_BYTE)v17 )
  {
    HIBYTE(v21) = 1;
    v11 = "missing required field 'expr'";
    goto LABEL_15;
  }
  v9 = *(__int64 **)a1;
  if ( a3 )
    v10 = sub_B0EF30(v9, v14, v16, 1u, 1);
  else
    v10 = sub_B0EF30(v9, v14, v16, 0, 1);
  *a2 = v10;
  return v8;
}

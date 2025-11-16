// Function: sub_121B840
// Address: 0x121b840
//
__int64 __fastcall sub_121B840(__int64 a1)
{
  _BYTE *v1; // rsi
  __int64 v2; // rdx
  unsigned int v3; // r12d
  __int64 v5; // r13
  int v6; // eax
  unsigned __int64 v7; // rsi
  __int64 v8; // [rsp+8h] [rbp-98h]
  __int64 v9; // [rsp+18h] [rbp-88h] BYREF
  _QWORD *v10[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v11[2]; // [rsp+30h] [rbp-70h] BYREF
  const char *v12; // [rsp+40h] [rbp-60h] BYREF
  char v13; // [rsp+60h] [rbp-40h]
  char v14; // [rsp+61h] [rbp-3Fh]

  v1 = *(_BYTE **)(a1 + 248);
  v2 = *(_QWORD *)(a1 + 256);
  v10[0] = v11;
  sub_12060D0((__int64 *)v10, v1, (__int64)&v1[v2]);
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 3, "expected '=' here")
    || (unsigned __int8)sub_120AFE0(a1, 14, "Expected '!' here")
    || (unsigned __int8)sub_120AFE0(a1, 8, "Expected '{' here") )
  {
LABEL_2:
    v3 = 1;
  }
  else
  {
    v5 = sub_BA8E40(*(_QWORD *)(a1 + 344), v10[0], (size_t)v10[1]);
    v6 = *(_DWORD *)(a1 + 240);
    if ( v6 != 9 )
    {
      v8 = a1 + 248;
      while ( 1 )
      {
        v9 = 0;
        if ( v6 == 511 )
        {
          if ( !(unsigned int)sub_2241AC0(v8, "DIExpression") )
          {
            if ( (unsigned __int8)sub_1210190(a1, &v9, 0) )
              goto LABEL_2;
            goto LABEL_13;
          }
          if ( *(_DWORD *)(a1 + 240) == 511 && !(unsigned int)sub_2241AC0(v8, "DIArgList") )
          {
            v14 = 1;
            v7 = *(_QWORD *)(a1 + 232);
            v13 = 3;
            v12 = "found DIArgList outside of function";
            sub_11FD800(a1 + 176, v7, (__int64)&v12, 1);
            goto LABEL_2;
          }
        }
        if ( (unsigned __int8)sub_120AFE0(a1, 14, "Expected '!' here") || (unsigned __int8)sub_121B570(a1, &v9) )
          goto LABEL_2;
LABEL_13:
        sub_B979A0(v5, v9);
        if ( *(_DWORD *)(a1 + 240) != 4 )
          break;
        v6 = sub_1205200(a1 + 176);
        *(_DWORD *)(a1 + 240) = v6;
      }
    }
    v3 = sub_120AFE0(a1, 9, "expected end of metadata node");
  }
  if ( v10[0] != v11 )
    j_j___libc_free_0(v10[0], v11[0] + 1LL);
  return v3;
}

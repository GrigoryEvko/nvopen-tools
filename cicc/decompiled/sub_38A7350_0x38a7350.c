// Function: sub_38A7350
// Address: 0x38a7350
//
__int64 __fastcall sub_38A7350(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v7; // r13
  int v8; // eax
  char v9; // al
  unsigned __int64 v10; // r15
  unsigned int v11; // r14d
  __int64 *v12; // rdi
  __int64 v13; // rax
  const char *v15; // rax
  int v16; // eax
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rsi
  __int64 v20; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v21; // [rsp+18h] [rbp-B8h]
  __int64 v22; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v23; // [rsp+28h] [rbp-A8h]
  __int64 v24; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v25; // [rsp+38h] [rbp-98h]
  __int64 v26; // [rsp+40h] [rbp-90h] BYREF
  __int16 v27; // [rsp+48h] [rbp-88h]
  __int64 v28; // [rsp+50h] [rbp-80h] BYREF
  __int16 v29; // [rsp+58h] [rbp-78h]
  _QWORD v30[2]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v31; // [rsp+70h] [rbp-60h]
  _QWORD v32[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v33; // [rsp+90h] [rbp-40h]

  v25 = 256;
  v7 = a1 + 8;
  v20 = 0;
  v21 = 256;
  v22 = 0;
  v23 = 256;
  v24 = 0;
  v26 = 0;
  v27 = 256;
  v28 = 0;
  v29 = 256;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v8 = *(_DWORD *)(a1 + 64);
  if ( v8 != 13 )
  {
    if ( v8 == 372 )
    {
      do
      {
        if ( sub_2241AC0(a1 + 72, "scope") )
        {
          if ( sub_2241AC0(a1 + 72, "name") )
          {
            if ( sub_2241AC0(a1 + 72, "configMacros") )
            {
              if ( sub_2241AC0(a1 + 72, "includePath") )
              {
                if ( sub_2241AC0(a1 + 72, "isysroot") )
                {
                  v18 = *(_QWORD *)(a1 + 56);
                  v30[1] = a1 + 72;
                  v30[0] = "invalid field '";
                  v31 = 1027;
                  v32[0] = v30;
                  v32[1] = "'";
                  v33 = 770;
                  v9 = sub_38814C0(v7, v18, (__int64)v32);
                }
                else
                {
                  v9 = sub_388B8F0(a1, (__int64)"isysroot", 8, (__int64)&v28);
                }
              }
              else
              {
                v9 = sub_388B8F0(a1, (__int64)"includePath", 11, (__int64)&v26);
              }
            }
            else
            {
              v9 = sub_388B8F0(a1, (__int64)"configMacros", 12, (__int64)&v24);
            }
          }
          else
          {
            v9 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v22);
          }
        }
        else
        {
          v9 = sub_38A29E0(a1, (__int64)"scope", 5, (__int64)&v20, a4, a5, a6);
        }
        if ( v9 )
          return 1;
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v16 = sub_3887100(v7);
        *(_DWORD *)(a1 + 64) = v16;
      }
      while ( v16 == 372 );
    }
    v17 = *(_QWORD *)(a1 + 56);
    v33 = 259;
    v32[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v7, v17, (__int64)v32) )
      return 1;
  }
LABEL_8:
  v10 = *(_QWORD *)(a1 + 56);
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v11 )
  {
    if ( (_BYTE)v21 )
    {
      if ( (_BYTE)v23 )
      {
        v12 = *(__int64 **)a1;
        if ( a3 )
          v13 = sub_15C1EB0(v12, v20, v22, v24, v26, v28, 1u, 1);
        else
          v13 = sub_15C1EB0(v12, v20, v22, v24, v26, v28, 0, 1);
        *a2 = v13;
        return v11;
      }
      HIBYTE(v33) = 1;
      v15 = "missing required field 'name'";
    }
    else
    {
      HIBYTE(v33) = 1;
      v15 = "missing required field 'scope'";
    }
    v32[0] = v15;
    LOBYTE(v33) = 3;
    return (unsigned int)sub_38814C0(v7, v10, (__int64)v32);
  }
  return v11;
}

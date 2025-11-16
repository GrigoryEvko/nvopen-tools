// Function: sub_38A8660
// Address: 0x38a8660
//
__int64 __fastcall sub_38A8660(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
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
  __int64 v27; // [rsp+48h] [rbp-88h]
  __int64 v28; // [rsp+50h] [rbp-80h]
  _QWORD v29[2]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v30; // [rsp+70h] [rbp-60h]
  _QWORD v31[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v32; // [rsp+90h] [rbp-40h]

  v25 = 256;
  v7 = a1 + 8;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 256;
  v24 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0xFFFFFFFFLL;
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
            if ( sub_2241AC0(a1 + 72, "file") )
            {
              if ( sub_2241AC0(a1 + 72, "line") )
              {
                v18 = *(_QWORD *)(a1 + 56);
                v29[1] = a1 + 72;
                v29[0] = "invalid field '";
                v30 = 1027;
                v31[0] = v29;
                v31[1] = "'";
                v32 = 770;
                v9 = sub_38814C0(v7, v18, (__int64)v31);
              }
              else
              {
                v9 = sub_38895C0(a1, (__int64)"line", 4, (__int64)&v26);
              }
            }
            else
            {
              v9 = sub_38A29E0(a1, (__int64)"file", 4, (__int64)&v24, a4, a5, a6);
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
    v32 = 259;
    v31[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v7, v17, (__int64)v31) )
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
        if ( (_BYTE)v25 )
        {
          if ( (_BYTE)v27 )
          {
            v12 = *(__int64 **)a1;
            if ( a3 )
              v13 = sub_15C3EA0(v12, v20, v22, v24, v26, 1u, 1);
            else
              v13 = sub_15C3EA0(v12, v20, v22, v24, v26, 0, 1);
            *a2 = v13;
            return v11;
          }
          HIBYTE(v32) = 1;
          v15 = "missing required field 'line'";
        }
        else
        {
          HIBYTE(v32) = 1;
          v15 = "missing required field 'file'";
        }
      }
      else
      {
        HIBYTE(v32) = 1;
        v15 = "missing required field 'name'";
      }
    }
    else
    {
      HIBYTE(v32) = 1;
      v15 = "missing required field 'scope'";
    }
    v31[0] = v15;
    LOBYTE(v32) = 3;
    return (unsigned int)sub_38814C0(v7, v10, (__int64)v31);
  }
  return v11;
}

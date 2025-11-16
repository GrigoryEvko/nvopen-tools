// Function: sub_38A9050
// Address: 0x38a9050
//
__int64 __fastcall sub_38A9050(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // r13
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
  __int64 v20; // [rsp+10h] [rbp-F0h] BYREF
  __int16 v21; // [rsp+18h] [rbp-E8h]
  __int64 v22; // [rsp+20h] [rbp-E0h] BYREF
  __int16 v23; // [rsp+28h] [rbp-D8h]
  __int64 v24; // [rsp+30h] [rbp-D0h] BYREF
  __int16 v25; // [rsp+38h] [rbp-C8h]
  __int64 v26; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v27; // [rsp+48h] [rbp-B8h]
  __int64 v28; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v29; // [rsp+58h] [rbp-A8h]
  __int64 v30; // [rsp+60h] [rbp-A0h]
  _QWORD v31[4]; // [rsp+70h] [rbp-90h] BYREF
  _QWORD v32[2]; // [rsp+90h] [rbp-70h] BYREF
  __int16 v33; // [rsp+A0h] [rbp-60h]
  _QWORD v34[2]; // [rsp+B0h] [rbp-50h] BYREF
  __int16 v35; // [rsp+C0h] [rbp-40h]

  v6 = a1 + 8;
  v23 = 256;
  v28 = 0;
  v29 = 0;
  v30 = 0xFFFF;
  v20 = 0;
  v21 = 256;
  v22 = 0;
  v24 = 0;
  v25 = 256;
  v31[0] = 0;
  v31[1] = 0;
  v31[2] = 0xFFFFFFFFLL;
  v26 = 0;
  v27 = 256;
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
        if ( sub_2241AC0(a1 + 72, "tag") )
        {
          if ( sub_2241AC0(a1 + 72, "scope") )
          {
            if ( sub_2241AC0(a1 + 72, "entity") )
            {
              if ( sub_2241AC0(a1 + 72, "file") )
              {
                if ( sub_2241AC0(a1 + 72, "line") )
                {
                  if ( sub_2241AC0(a1 + 72, "name") )
                  {
                    v18 = *(_QWORD *)(a1 + 56);
                    v32[1] = a1 + 72;
                    v35 = 770;
                    v32[0] = "invalid field '";
                    v33 = 1027;
                    v34[0] = v32;
                    v34[1] = "'";
                    v9 = sub_38814C0(v6, v18, (__int64)v34);
                  }
                  else
                  {
                    v9 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v26);
                  }
                }
                else
                {
                  v9 = sub_38895C0(a1, (__int64)"line", 4, (__int64)v31);
                }
              }
              else
              {
                v9 = sub_38A29E0(a1, (__int64)"file", 4, (__int64)&v24, a4, a5, a6);
              }
            }
            else
            {
              v9 = sub_38A29E0(a1, (__int64)"entity", 6, (__int64)&v22, a4, a5, a6);
            }
          }
          else
          {
            v9 = sub_38A29E0(a1, (__int64)"scope", 5, (__int64)&v20, a4, a5, a6);
          }
        }
        else
        {
          v9 = sub_38899B0(a1, (__int64)"tag", 3, (__int64)&v28);
        }
        if ( v9 )
          return 1;
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v16 = sub_3887100(v6);
        *(_DWORD *)(a1 + 64) = v16;
      }
      while ( v16 == 372 );
    }
    v17 = *(_QWORD *)(a1 + 56);
    v35 = 259;
    v34[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v6, v17, (__int64)v34) )
      return 1;
  }
LABEL_8:
  v10 = *(_QWORD *)(a1 + 56);
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v11 )
  {
    if ( (_BYTE)v29 )
    {
      if ( (_BYTE)v21 )
      {
        v12 = *(__int64 **)a1;
        if ( a3 )
          v13 = sub_15C6270(v12, v28, v20, v22, v24, v31[0], v26, 1u, 1);
        else
          v13 = sub_15C6270(v12, v28, v20, v22, v24, v31[0], v26, 0, 1);
        *a2 = v13;
        return v11;
      }
      HIBYTE(v35) = 1;
      v15 = "missing required field 'scope'";
    }
    else
    {
      HIBYTE(v35) = 1;
      v15 = "missing required field 'tag'";
    }
    v34[0] = v15;
    LOBYTE(v35) = 3;
    return (unsigned int)sub_38814C0(v6, v10, (__int64)v34);
  }
  return v11;
}

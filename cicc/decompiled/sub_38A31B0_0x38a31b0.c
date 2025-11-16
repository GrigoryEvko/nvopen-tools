// Function: sub_38A31B0
// Address: 0x38a31b0
//
__int64 __fastcall sub_38A31B0(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // r13
  int v9; // eax
  char v10; // al
  unsigned int v11; // r13d
  __int64 *v12; // rdi
  __int64 v13; // rax
  int v15; // eax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rsi
  __int64 v18; // [rsp+10h] [rbp-120h] BYREF
  __int16 v19; // [rsp+18h] [rbp-118h]
  __int64 v20; // [rsp+20h] [rbp-110h] BYREF
  __int16 v21; // [rsp+28h] [rbp-108h]
  __int64 v22; // [rsp+30h] [rbp-100h] BYREF
  __int16 v23; // [rsp+38h] [rbp-F8h]
  _QWORD v24[4]; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v25[4]; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD v26[4]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD v27[4]; // [rsp+A0h] [rbp-90h] BYREF
  _QWORD v28[2]; // [rsp+C0h] [rbp-70h] BYREF
  __int16 v29; // [rsp+D0h] [rbp-60h]
  _QWORD v30[2]; // [rsp+E0h] [rbp-50h] BYREF
  __int16 v31; // [rsp+F0h] [rbp-40h]

  v6 = a1 + 8;
  v21 = 256;
  v24[0] = 18;
  v24[1] = 0;
  v24[2] = 0xFFFF;
  v18 = 0;
  v19 = 256;
  v20 = 0;
  v22 = 0;
  v23 = 256;
  v25[0] = 0;
  v25[1] = 0;
  v25[2] = -1;
  v26[0] = 0;
  v26[1] = 0;
  v26[2] = 0xFFFFFFFFLL;
  v27[0] = 0;
  v27[1] = 0;
  v27[2] = 255;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v9 = *(_DWORD *)(a1 + 64);
  if ( v9 != 13 )
  {
    if ( v9 == 372 )
    {
      do
      {
        if ( sub_2241AC0(a1 + 72, "tag") )
        {
          if ( sub_2241AC0(a1 + 72, "name") )
          {
            if ( sub_2241AC0(a1 + 72, "stringLength") )
            {
              if ( sub_2241AC0(a1 + 72, "stringLengthExpression") )
              {
                if ( sub_2241AC0(a1 + 72, "size") )
                {
                  if ( sub_2241AC0(a1 + 72, "align") )
                  {
                    if ( sub_2241AC0(a1 + 72, "encoding") )
                    {
                      v17 = *(_QWORD *)(a1 + 56);
                      v28[1] = a1 + 72;
                      v31 = 770;
                      v28[0] = "invalid field '";
                      v29 = 1027;
                      v30[0] = v28;
                      v30[1] = "'";
                      v10 = sub_38814C0(v6, v17, (__int64)v30);
                    }
                    else
                    {
                      v10 = sub_3889670(a1, (__int64)"encoding", 8, (__int64)v27);
                    }
                  }
                  else
                  {
                    v10 = sub_3889510(a1, (__int64)"align", 5, (__int64)v26);
                  }
                }
                else
                {
                  v10 = sub_3889510(a1, (__int64)"size", 4, (__int64)v25);
                }
              }
              else
              {
                v10 = sub_38A29E0(a1, (__int64)"stringLengthExpression", 22, (__int64)&v22, a4, a5, a6);
              }
            }
            else
            {
              v10 = sub_38A29E0(a1, (__int64)"stringLength", 12, (__int64)&v20, a4, a5, a6);
            }
          }
          else
          {
            v10 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v18);
          }
        }
        else
        {
          v10 = sub_38899B0(a1, (__int64)"tag", 3, (__int64)v24);
        }
        if ( v10 )
          return 1;
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v15 = sub_3887100(v6);
        *(_DWORD *)(a1 + 64) = v15;
      }
      while ( v15 == 372 );
    }
    v16 = *(_QWORD *)(a1 + 56);
    v31 = 259;
    v30[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v6, v16, (__int64)v30) )
      return 1;
  }
LABEL_8:
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( (_BYTE)v11 )
    return 1;
  v12 = *(__int64 **)a1;
  if ( a3 )
    v13 = sub_15BCE80(v12, v24[0], v18, v20, v22, v25[0], v26[0], v27[0], 1u, 1);
  else
    v13 = sub_15BCE80(v12, v24[0], v18, v20, v22, v25[0], v26[0], v27[0], 0, 1);
  *a2 = v13;
  return v11;
}

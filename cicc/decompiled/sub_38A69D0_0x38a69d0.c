// Function: sub_38A69D0
// Address: 0x38a69d0
//
__int64 __fastcall sub_38A69D0(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v7; // r13
  int v8; // eax
  char v9; // al
  unsigned __int64 v10; // r14
  unsigned int v11; // r15d
  __int64 *v12; // rdi
  __int64 v13; // rax
  int v15; // eax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rsi
  __int64 v19; // [rsp+10h] [rbp-D0h] BYREF
  __int16 v20; // [rsp+18h] [rbp-C8h]
  __int64 v21; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v22; // [rsp+28h] [rbp-B8h]
  __int64 v23; // [rsp+30h] [rbp-B0h] BYREF
  __int16 v24; // [rsp+38h] [rbp-A8h]
  __int64 v25; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v26; // [rsp+48h] [rbp-98h]
  _QWORD v27[4]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v28[2]; // [rsp+70h] [rbp-70h] BYREF
  __int16 v29; // [rsp+80h] [rbp-60h]
  _QWORD v30[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v31; // [rsp+A0h] [rbp-40h]

  v24 = 256;
  v7 = a1 + 8;
  v19 = 0;
  v20 = 256;
  v21 = 0;
  v22 = 256;
  v23 = 0;
  v25 = 0;
  v26 = 256;
  v27[0] = 0;
  v27[1] = 0;
  v27[2] = 0xFFFFFFFFLL;
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
          if ( sub_2241AC0(a1 + 72, "declaration") )
          {
            if ( sub_2241AC0(a1 + 72, "name") )
            {
              if ( sub_2241AC0(a1 + 72, "file") )
              {
                if ( sub_2241AC0(a1 + 72, "line") )
                {
                  v17 = *(_QWORD *)(a1 + 56);
                  v28[1] = a1 + 72;
                  v28[0] = "invalid field '";
                  v29 = 1027;
                  v30[0] = v28;
                  v30[1] = "'";
                  v31 = 770;
                  v9 = sub_38814C0(v7, v17, (__int64)v30);
                }
                else
                {
                  v9 = sub_38895C0(a1, (__int64)"line", 4, (__int64)v27);
                }
              }
              else
              {
                v9 = sub_38A29E0(a1, (__int64)"file", 4, (__int64)&v25, a4, a5, a6);
              }
            }
            else
            {
              v9 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v23);
            }
          }
          else
          {
            v9 = sub_38A29E0(a1, (__int64)"declaration", 11, (__int64)&v21, a4, a5, a6);
          }
        }
        else
        {
          v9 = sub_38A29E0(a1, (__int64)"scope", 5, (__int64)&v19, a4, a5, a6);
        }
        if ( v9 )
          return 1;
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v15 = sub_3887100(v7);
        *(_DWORD *)(a1 + 64) = v15;
      }
      while ( v15 == 372 );
    }
    v16 = *(_QWORD *)(a1 + 56);
    v31 = 259;
    v30[0] = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v7, v16, (__int64)v30) )
      return 1;
  }
LABEL_8:
  v10 = *(_QWORD *)(a1 + 56);
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v11 )
  {
    if ( (_BYTE)v20 )
    {
      v12 = *(__int64 **)a1;
      if ( a3 )
        v13 = sub_15C1830(v12, v19, v21, v23, v25, v27[0], 1u, 1);
      else
        v13 = sub_15C1830(v12, v19, v21, v23, v25, v27[0], 0, 1);
      *a2 = v13;
    }
    else
    {
      v31 = 259;
      v30[0] = "missing required field 'scope'";
      return (unsigned int)sub_38814C0(v7, v10, (__int64)v30);
    }
  }
  return v11;
}

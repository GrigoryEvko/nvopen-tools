// Function: sub_38A2DD0
// Address: 0x38a2dd0
//
__int64 __fastcall sub_38A2DD0(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // r13
  int v9; // eax
  char v10; // al
  unsigned int v11; // r13d
  char v12; // cl
  __int64 v13; // rax
  int v15; // eax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rsi
  __int64 v18; // [rsp+10h] [rbp-F0h] BYREF
  __int16 v19; // [rsp+18h] [rbp-E8h]
  __int64 v20; // [rsp+20h] [rbp-E0h] BYREF
  __int16 v21; // [rsp+28h] [rbp-D8h]
  __int64 v22; // [rsp+30h] [rbp-D0h] BYREF
  __int16 v23; // [rsp+38h] [rbp-C8h]
  __int64 v24; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v25; // [rsp+48h] [rbp-B8h]
  _QWORD v26[2]; // [rsp+50h] [rbp-B0h] BYREF
  __int16 v27; // [rsp+60h] [rbp-A0h]
  _QWORD v28[2]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v29; // [rsp+80h] [rbp-80h]
  __int64 v30[4]; // [rsp+90h] [rbp-70h] BYREF
  __int64 v31; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v32; // [rsp+B8h] [rbp-48h]
  unsigned __int64 v33; // [rsp+C0h] [rbp-40h]
  __int64 v34; // [rsp+C8h] [rbp-38h]

  v6 = a1 + 8;
  v30[2] = 0x8000000000000000LL;
  v33 = 0x8000000000000000LL;
  v21 = 256;
  v30[0] = 0;
  v30[1] = 0;
  v30[3] = 0x7FFFFFFFFFFFFFFFLL;
  v31 = 0;
  v32 = 0;
  v34 = 0x7FFFFFFFFFFFFFFFLL;
  v18 = 0;
  v19 = 256;
  v20 = 0;
  v22 = 0;
  v23 = 256;
  v24 = 0;
  v25 = 256;
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
        if ( sub_2241AC0(a1 + 72, "constLowerBound") )
        {
          if ( sub_2241AC0(a1 + 72, "constUpperBound") )
          {
            if ( sub_2241AC0(a1 + 72, "lowerBound") )
            {
              if ( sub_2241AC0(a1 + 72, "lowerBoundExpression") )
              {
                if ( sub_2241AC0(a1 + 72, "upperBound") )
                {
                  if ( sub_2241AC0(a1 + 72, "upperBoundExpression") )
                  {
                    v17 = *(_QWORD *)(a1 + 56);
                    v26[0] = "invalid field '";
                    v27 = 1027;
                    v29 = 770;
                    v28[0] = v26;
                    v26[1] = a1 + 72;
                    v28[1] = "'";
                    v10 = sub_38814C0(v6, v17, (__int64)v28);
                  }
                  else
                  {
                    v10 = sub_38A29E0(a1, (__int64)"upperBoundExpression", 20, (__int64)&v24, a4, a5, a6);
                  }
                }
                else
                {
                  v10 = sub_38A29E0(a1, (__int64)"upperBound", 10, (__int64)&v22, a4, a5, a6);
                }
              }
              else
              {
                v10 = sub_38A29E0(a1, (__int64)"lowerBoundExpression", 20, (__int64)&v20, a4, a5, a6);
              }
            }
            else
            {
              v10 = sub_38A29E0(a1, (__int64)"lowerBound", 10, (__int64)&v18, a4, a5, a6);
            }
          }
          else
          {
            v10 = sub_388AAB0(a1, (__int64)"constUpperBound", 15, (__int64)&v31);
          }
        }
        else
        {
          v10 = sub_388AAB0(a1, (__int64)"constLowerBound", 15, (__int64)v30);
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
    v28[0] = "expected field label here";
    v29 = 259;
    if ( (unsigned __int8)sub_38814C0(v6, v16, (__int64)v28) )
      return 1;
  }
LABEL_8:
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( (_BYTE)v11 )
    return 1;
  v12 = 0;
  if ( a3 )
  {
    if ( !(_BYTE)v32 )
      v12 = v23 ^ 1;
    v13 = sub_15BBC00(*(__int64 **)a1, v30[0], v31, v12, v18, v20, v22, v24, 1u, 1);
  }
  else
  {
    if ( !(_BYTE)v32 )
      v12 = v23 ^ 1;
    v13 = sub_15BBC00(*(__int64 **)a1, v30[0], v31, v12, v18, v20, v22, v24, 0, 1);
  }
  *a2 = v13;
  return v11;
}

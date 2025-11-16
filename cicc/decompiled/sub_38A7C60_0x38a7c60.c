// Function: sub_38A7C60
// Address: 0x38a7c60
//
__int64 __fastcall sub_38A7C60(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // r15
  int v9; // eax
  char v10; // al
  unsigned int v11; // r12d
  __int64 *v12; // rdi
  __int64 v13; // rax
  int v15; // eax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // [rsp+18h] [rbp-188h]
  __int16 v18; // [rsp+2Ch] [rbp-174h] BYREF
  __int16 v19; // [rsp+2Eh] [rbp-172h] BYREF
  __int64 v20; // [rsp+30h] [rbp-170h] BYREF
  __int16 v21; // [rsp+38h] [rbp-168h]
  __int64 v22; // [rsp+40h] [rbp-160h] BYREF
  __int16 v23; // [rsp+48h] [rbp-158h]
  __int64 v24; // [rsp+50h] [rbp-150h] BYREF
  __int16 v25; // [rsp+58h] [rbp-148h]
  __int64 v26; // [rsp+60h] [rbp-140h] BYREF
  __int16 v27; // [rsp+68h] [rbp-138h]
  __int64 v28; // [rsp+70h] [rbp-130h] BYREF
  __int16 v29; // [rsp+78h] [rbp-128h]
  __int64 v30; // [rsp+80h] [rbp-120h] BYREF
  __int16 v31; // [rsp+88h] [rbp-118h]
  _QWORD v32[4]; // [rsp+90h] [rbp-110h] BYREF
  _QWORD v33[4]; // [rsp+B0h] [rbp-F0h] BYREF
  __m128i v34; // [rsp+D0h] [rbp-D0h] BYREF
  char v35; // [rsp+E0h] [rbp-C0h]
  char v36; // [rsp+E1h] [rbp-BFh]
  __m128i v37; // [rsp+F0h] [rbp-B0h] BYREF
  __int16 v38; // [rsp+100h] [rbp-A0h]
  __m128i v39[2]; // [rsp+110h] [rbp-90h] BYREF
  __m128i v40; // [rsp+130h] [rbp-70h] BYREF
  char v41; // [rsp+140h] [rbp-60h]
  char v42; // [rsp+141h] [rbp-5Fh]
  __m128i v43; // [rsp+150h] [rbp-50h] BYREF
  char v44; // [rsp+160h] [rbp-40h]
  char v45; // [rsp+161h] [rbp-3Fh]

  v6 = a1 + 8;
  v21 = 0;
  v23 = 256;
  v27 = 256;
  v20 = 0;
  v22 = 0;
  v24 = 0;
  v25 = 256;
  v26 = 0;
  v32[0] = 0;
  v32[1] = 0;
  v32[2] = 0xFFFFFFFFLL;
  v28 = 0;
  v29 = 256;
  v18 = 0;
  v19 = 1;
  v30 = 0;
  v31 = 256;
  v33[0] = 0;
  v33[1] = 0;
  v33[2] = 0xFFFFFFFFLL;
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
        if ( sub_2241AC0(a1 + 72, "name") )
        {
          if ( sub_2241AC0(a1 + 72, "scope") )
          {
            if ( sub_2241AC0(a1 + 72, "linkageName") )
            {
              if ( sub_2241AC0(a1 + 72, "file") )
              {
                if ( sub_2241AC0(a1 + 72, "line") )
                {
                  if ( sub_2241AC0(a1 + 72, "type") )
                  {
                    if ( sub_2241AC0(a1 + 72, "isLocal") )
                    {
                      if ( sub_2241AC0(a1 + 72, "isDefinition") )
                      {
                        if ( sub_2241AC0(a1 + 72, "declaration") )
                        {
                          if ( sub_2241AC0(a1 + 72, "align") )
                          {
                            v42 = 1;
                            v40.m128i_i64[0] = (__int64)"'";
                            v34.m128i_i64[0] = (__int64)"invalid field '";
                            v41 = 3;
                            v38 = 260;
                            v37.m128i_i64[0] = a1 + 72;
                            v36 = 1;
                            v35 = 3;
                            sub_14EC200(v39, &v34, &v37);
                            sub_14EC200(&v43, v39, &v40);
                            v10 = sub_38814C0(v6, *(_QWORD *)(a1 + 56), (__int64)&v43);
                          }
                          else
                          {
                            v10 = sub_3889510(a1, (__int64)"align", 5, (__int64)v33);
                          }
                        }
                        else
                        {
                          v10 = sub_38A29E0(a1, (__int64)"declaration", 11, (__int64)&v30, a4, a5, a6);
                        }
                      }
                      else
                      {
                        v10 = sub_3887760(a1, (__int64)"isDefinition", 12, (__int64)&v19);
                      }
                    }
                    else
                    {
                      v10 = sub_3887760(a1, (__int64)"isLocal", 7, (__int64)&v18);
                    }
                  }
                  else
                  {
                    v10 = sub_38A29E0(a1, (__int64)"type", 4, (__int64)&v28, a4, a5, a6);
                  }
                }
                else
                {
                  v10 = sub_38895C0(a1, (__int64)"line", 4, (__int64)v32);
                }
              }
              else
              {
                v10 = sub_38A29E0(a1, (__int64)"file", 4, (__int64)&v26, a4, a5, a6);
              }
            }
            else
            {
              v10 = sub_388B8F0(a1, (__int64)"linkageName", 11, (__int64)&v24);
            }
          }
          else
          {
            v10 = sub_38A29E0(a1, (__int64)"scope", 5, (__int64)&v22, a4, a5, a6);
          }
        }
        else
        {
          v10 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v20);
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
    v45 = 1;
    v44 = 3;
    v43.m128i_i64[0] = (__int64)"expected field label here";
    if ( (unsigned __int8)sub_38814C0(v6, v16, (__int64)&v43) )
      return 1;
  }
LABEL_8:
  v17 = *(_QWORD *)(a1 + 56);
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v11 )
  {
    if ( (_BYTE)v21 )
    {
      v12 = *(__int64 **)a1;
      if ( a3 )
        v13 = sub_15C2FB0(v12, v22, v20, v24, v26, v32[0], v28, v18, v19, v30, v33[0], 1u, 1);
      else
        v13 = sub_15C2FB0(v12, v22, v20, v24, v26, v32[0], v28, v18, v19, v30, v33[0], 0, 1);
      *a2 = v13;
    }
    else
    {
      v45 = 1;
      v44 = 3;
      v43.m128i_i64[0] = (__int64)"missing required field 'name'";
      return (unsigned int)sub_38814C0(v6, v17, (__int64)&v43);
    }
  }
  return v11;
}

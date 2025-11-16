// Function: sub_38A4580
// Address: 0x38a4580
//
__int64 __fastcall sub_38A4580(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // r12
  int v7; // eax
  char v8; // al
  unsigned int v9; // r12d
  __int64 *v10; // rdi
  __int64 v11; // rax
  int v13; // eax
  unsigned __int64 v14; // rsi
  int v17; // [rsp+28h] [rbp-1C8h] BYREF
  char v18; // [rsp+2Ch] [rbp-1C4h]
  __int64 v19; // [rsp+30h] [rbp-1C0h] BYREF
  __int16 v20; // [rsp+38h] [rbp-1B8h]
  __int64 v21; // [rsp+40h] [rbp-1B0h] BYREF
  __int16 v22; // [rsp+48h] [rbp-1A8h]
  __int64 v23; // [rsp+50h] [rbp-1A0h] BYREF
  __int16 v24; // [rsp+58h] [rbp-198h]
  __int64 v25; // [rsp+60h] [rbp-190h] BYREF
  __int16 v26; // [rsp+68h] [rbp-188h]
  __int64 v27; // [rsp+70h] [rbp-180h] BYREF
  __int16 v28; // [rsp+78h] [rbp-178h]
  _QWORD v29[4]; // [rsp+80h] [rbp-170h] BYREF
  _QWORD v30[4]; // [rsp+A0h] [rbp-150h] BYREF
  __int64 v31[4]; // [rsp+C0h] [rbp-130h] BYREF
  _QWORD v32[4]; // [rsp+E0h] [rbp-110h] BYREF
  __int64 v33[4]; // [rsp+100h] [rbp-F0h] BYREF
  __m128i v34; // [rsp+120h] [rbp-D0h] BYREF
  char v35; // [rsp+130h] [rbp-C0h]
  char v36; // [rsp+131h] [rbp-BFh]
  __m128i v37; // [rsp+140h] [rbp-B0h] BYREF
  __int16 v38; // [rsp+150h] [rbp-A0h]
  __m128i v39[2]; // [rsp+160h] [rbp-90h] BYREF
  __m128i v40; // [rsp+180h] [rbp-70h] BYREF
  char v41; // [rsp+190h] [rbp-60h]
  char v42; // [rsp+191h] [rbp-5Fh]
  __m128i v43; // [rsp+1A0h] [rbp-50h] BYREF
  char v44; // [rsp+1B0h] [rbp-40h]
  char v45; // [rsp+1B1h] [rbp-3Fh]

  v6 = a1 + 8;
  v20 = 256;
  v22 = 256;
  v26 = 256;
  v30[2] = 0xFFFFFFFFLL;
  v24 = 256;
  v32[2] = 0xFFFFFFFFLL;
  v29[0] = 1;
  v29[1] = 0;
  v29[2] = 0xFFFF;
  v19 = 0;
  v21 = 0;
  v30[0] = 0;
  v30[1] = 0;
  v23 = 0;
  v25 = 0;
  v31[0] = 0;
  v31[1] = 0;
  v31[2] = -1;
  v32[0] = 0;
  v32[1] = 0;
  v33[0] = 0;
  v33[1] = 0;
  v33[2] = -1;
  v17 = 0;
  v18 = 0;
  v27 = 0;
  v28 = 256;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v7 = *(_DWORD *)(a1 + 64);
  if ( v7 != 13 )
  {
    if ( v7 == 372 )
    {
      do
      {
        if ( sub_2241AC0(a1 + 72, "tag") )
        {
          if ( sub_2241AC0(a1 + 72, "name") )
          {
            if ( sub_2241AC0(a1 + 72, "file") )
            {
              if ( sub_2241AC0(a1 + 72, "line") )
              {
                if ( sub_2241AC0(a1 + 72, "scope") )
                {
                  if ( sub_2241AC0(a1 + 72, "baseType") )
                  {
                    if ( sub_2241AC0(a1 + 72, "size") )
                    {
                      if ( sub_2241AC0(a1 + 72, "align") )
                      {
                        if ( sub_2241AC0(a1 + 72, "offset") )
                        {
                          if ( sub_2241AC0(a1 + 72, "flags") )
                          {
                            if ( sub_2241AC0(a1 + 72, "elements") )
                            {
                              v40.m128i_i64[0] = (__int64)"'";
                              v34.m128i_i64[0] = (__int64)"invalid field '";
                              v42 = 1;
                              v41 = 3;
                              v38 = 260;
                              v37.m128i_i64[0] = a1 + 72;
                              v36 = 1;
                              v35 = 3;
                              sub_14EC200(v39, &v34, &v37);
                              sub_14EC200(&v43, v39, &v40);
                              v8 = sub_38814C0(v6, *(_QWORD *)(a1 + 56), (__int64)&v43);
                            }
                            else
                            {
                              v8 = sub_38A29E0(a1, (__int64)"elements", 8, (__int64)&v27, a4, a5, a6);
                            }
                          }
                          else
                          {
                            v8 = sub_388BBA0(a1, (__int64)"flags", 5, (__int64)&v17);
                          }
                        }
                        else
                        {
                          v8 = sub_3889510(a1, (__int64)"offset", 6, (__int64)v33);
                        }
                      }
                      else
                      {
                        v8 = sub_3889510(a1, (__int64)"align", 5, (__int64)v32);
                      }
                    }
                    else
                    {
                      v8 = sub_3889510(a1, (__int64)"size", 4, (__int64)v31);
                    }
                  }
                  else
                  {
                    v8 = sub_38A29E0(a1, (__int64)"baseType", 8, (__int64)&v25, a4, a5, a6);
                  }
                }
                else
                {
                  v8 = sub_38A29E0(a1, (__int64)"scope", 5, (__int64)&v23, a4, a5, a6);
                }
              }
              else
              {
                v8 = sub_38895C0(a1, (__int64)"line", 4, (__int64)v30);
              }
            }
            else
            {
              v8 = sub_38A29E0(a1, (__int64)"file", 4, (__int64)&v21, a4, a5, a6);
            }
          }
          else
          {
            v8 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v19);
          }
        }
        else
        {
          v8 = sub_38899B0(a1, (__int64)"tag", 3, (__int64)v29);
        }
        if ( v8 )
          return 1;
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v13 = sub_3887100(v6);
        *(_DWORD *)(a1 + 64) = v13;
      }
      while ( v13 == 372 );
    }
    v14 = *(_QWORD *)(a1 + 56);
    v45 = 1;
    v44 = 3;
    v43.m128i_i64[0] = (__int64)"expected field label here";
    if ( (unsigned __int8)sub_38814C0(v6, v14, (__int64)&v43) )
      return 1;
  }
LABEL_8:
  v9 = sub_388AF10(a1, 13, "expected ')' here");
  if ( (_BYTE)v9 )
    return 1;
  v10 = *(__int64 **)a1;
  if ( a3 )
    v11 = sub_15BE7F0(v10, v29[0], v19, v21, v30[0], v23, v25, v31[0], v32[0], v33[0], v17, v27, 1u, 1);
  else
    v11 = sub_15BE7F0(v10, v29[0], v19, v21, v30[0], v23, v25, v31[0], v32[0], v33[0], v17, v27, 0, 1);
  *a2 = v11;
  return v9;
}

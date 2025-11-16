// Function: sub_38A35C0
// Address: 0x38a35c0
//
__int64 __fastcall sub_38A35C0(__int64 a1, __int64 *a2, __int8 a3, double a4, double a5, double a6)
{
  __int8 v6; // r15
  __int64 v7; // r13
  int v9; // eax
  char v10; // al
  unsigned int v11; // r14d
  __int8 v12; // cl
  _QWORD *v13; // rdi
  __int64 v14; // rax
  const char *v16; // rax
  int v17; // eax
  unsigned __int64 v18; // rsi
  _QWORD *v19; // rdi
  __int32 v20; // [rsp+14h] [rbp-20Ch]
  unsigned __int64 v21; // [rsp+20h] [rbp-200h]
  int v23; // [rsp+38h] [rbp-1E8h] BYREF
  char v24; // [rsp+3Ch] [rbp-1E4h]
  __int64 v25; // [rsp+40h] [rbp-1E0h] BYREF
  __int16 v26; // [rsp+48h] [rbp-1D8h]
  __int64 v27; // [rsp+50h] [rbp-1D0h] BYREF
  __int16 v28; // [rsp+58h] [rbp-1C8h]
  __int64 v29; // [rsp+60h] [rbp-1C0h] BYREF
  __int16 v30; // [rsp+68h] [rbp-1B8h]
  __int64 v31; // [rsp+70h] [rbp-1B0h] BYREF
  __int16 v32; // [rsp+78h] [rbp-1A8h]
  __int64 v33; // [rsp+80h] [rbp-1A0h] BYREF
  __int16 v34; // [rsp+88h] [rbp-198h]
  __int64 v35; // [rsp+90h] [rbp-190h] BYREF
  __int64 v36; // [rsp+98h] [rbp-188h]
  __int64 v37; // [rsp+A0h] [rbp-180h]
  _QWORD v38[4]; // [rsp+B0h] [rbp-170h] BYREF
  __int64 v39[4]; // [rsp+D0h] [rbp-150h] BYREF
  _QWORD v40[4]; // [rsp+F0h] [rbp-130h] BYREF
  __int64 v41[4]; // [rsp+110h] [rbp-110h] BYREF
  _QWORD v42[4]; // [rsp+130h] [rbp-F0h] BYREF
  __m128i v43; // [rsp+150h] [rbp-D0h] BYREF
  char v44; // [rsp+160h] [rbp-C0h]
  char v45; // [rsp+161h] [rbp-BFh]
  __m128i v46; // [rsp+170h] [rbp-B0h] BYREF
  __int16 v47; // [rsp+180h] [rbp-A0h]
  __m128i v48[2]; // [rsp+190h] [rbp-90h] BYREF
  __m128i v49; // [rsp+1B0h] [rbp-70h] BYREF
  char v50; // [rsp+1C0h] [rbp-60h]
  char v51; // [rsp+1C1h] [rbp-5Fh]
  __m128i v52; // [rsp+1D0h] [rbp-50h] BYREF
  char v53; // [rsp+1E0h] [rbp-40h]
  char v54; // [rsp+1E1h] [rbp-3Fh]

  v6 = a3;
  v7 = a1 + 8;
  v26 = 256;
  v28 = 256;
  v32 = 256;
  v38[2] = 0xFFFFFFFFLL;
  v30 = 256;
  v40[2] = 0xFFFFFFFFLL;
  v35 = 0;
  v36 = 0;
  v37 = 0xFFFF;
  v25 = 0;
  v27 = 0;
  v38[0] = 0;
  v38[1] = 0;
  v29 = 0;
  v31 = 0;
  v39[0] = 0;
  v39[1] = 0;
  v39[2] = -1;
  v40[0] = 0;
  v40[1] = 0;
  v41[0] = 0;
  v41[1] = 0;
  v41[2] = -1;
  v23 = 0;
  v24 = 0;
  v33 = 0;
  v34 = 256;
  v42[0] = 0xFFFFFFFFLL;
  v42[1] = 0;
  v42[2] = 0xFFFFFFFFLL;
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
                            if ( sub_2241AC0(a1 + 72, "extraData") )
                            {
                              if ( sub_2241AC0(a1 + 72, "dwarfAddressSpace") )
                              {
                                v51 = 1;
                                v49.m128i_i64[0] = (__int64)"'";
                                v43.m128i_i64[0] = (__int64)"invalid field '";
                                v50 = 3;
                                v47 = 260;
                                v46.m128i_i64[0] = a1 + 72;
                                v45 = 1;
                                v44 = 3;
                                sub_14EC200(v48, &v43, &v46);
                                sub_14EC200(&v52, v48, &v49);
                                v10 = sub_38814C0(v7, *(_QWORD *)(a1 + 56), (__int64)&v52);
                              }
                              else
                              {
                                v10 = sub_3889510(a1, (__int64)"dwarfAddressSpace", 17, (__int64)v42);
                              }
                            }
                            else
                            {
                              v10 = sub_38A29E0(a1, (__int64)"extraData", 9, (__int64)&v33, a4, a5, a6);
                            }
                          }
                          else
                          {
                            v10 = sub_388BBA0(a1, (__int64)"flags", 5, (__int64)&v23);
                          }
                        }
                        else
                        {
                          v10 = sub_3889510(a1, (__int64)"offset", 6, (__int64)v41);
                        }
                      }
                      else
                      {
                        v10 = sub_3889510(a1, (__int64)"align", 5, (__int64)v40);
                      }
                    }
                    else
                    {
                      v10 = sub_3889510(a1, (__int64)"size", 4, (__int64)v39);
                    }
                  }
                  else
                  {
                    v10 = sub_38A29E0(a1, (__int64)"baseType", 8, (__int64)&v31, a4, a5, a6);
                  }
                }
                else
                {
                  v10 = sub_38A29E0(a1, (__int64)"scope", 5, (__int64)&v29, a4, a5, a6);
                }
              }
              else
              {
                v10 = sub_38895C0(a1, (__int64)"line", 4, (__int64)v38);
              }
            }
            else
            {
              v10 = sub_38A29E0(a1, (__int64)"file", 4, (__int64)&v27, a4, a5, a6);
            }
          }
          else
          {
            v10 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v25);
          }
        }
        else
        {
          v10 = sub_38899B0(a1, (__int64)"tag", 3, (__int64)&v35);
        }
        if ( v10 )
          return 1;
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v17 = sub_3887100(v7);
        *(_DWORD *)(a1 + 64) = v17;
      }
      while ( v17 == 372 );
    }
    v18 = *(_QWORD *)(a1 + 56);
    v54 = 1;
    v53 = 3;
    v52.m128i_i64[0] = (__int64)"expected field label here";
    if ( (unsigned __int8)sub_38814C0(v7, v18, (__int64)&v52) )
      return 1;
  }
LABEL_8:
  v21 = *(_QWORD *)(a1 + 56);
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v11 )
  {
    if ( (_BYTE)v36 )
    {
      v12 = v32;
      if ( (_BYTE)v32 )
      {
        if ( v42[0] == 0xFFFFFFFFLL )
        {
          v6 = 0;
          v12 = 0;
          if ( !a3 )
            goto LABEL_13;
        }
        else
        {
          v20 = v42[0];
          if ( !a3 )
          {
LABEL_13:
            v52.m128i_i8[4] = v12;
            v13 = *(_QWORD **)a1;
            if ( v12 )
              v52.m128i_i32[0] = v20;
            v14 = sub_15BD310(
                    v13,
                    v35,
                    v25,
                    v27,
                    v38[0],
                    v29,
                    v31,
                    v39[0],
                    v40[0],
                    v41[0],
                    v52.m128i_i32,
                    v23,
                    v33,
                    0,
                    1);
LABEL_16:
            *a2 = v14;
            return v11;
          }
        }
        v52.m128i_i8[4] = v6;
        v19 = *(_QWORD **)a1;
        if ( v6 )
          v52.m128i_i32[0] = v20;
        v14 = sub_15BD310(v19, v35, v25, v27, v38[0], v29, v31, v39[0], v40[0], v41[0], v52.m128i_i32, v23, v33, 1u, 1);
        goto LABEL_16;
      }
      v54 = 1;
      v16 = "missing required field 'baseType'";
    }
    else
    {
      v54 = 1;
      v16 = "missing required field 'tag'";
    }
    v52.m128i_i64[0] = (__int64)v16;
    v53 = 3;
    return (unsigned int)sub_38814C0(v7, v21, (__int64)&v52);
  }
  return v11;
}

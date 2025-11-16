// Function: sub_38A3D00
// Address: 0x38a3d00
//
__int64 __fastcall sub_38A3D00(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v7; // r12
  int v8; // eax
  char v9; // al
  unsigned __int64 v10; // r13
  unsigned int v11; // r14d
  __int64 v12; // rsi
  __int64 v13; // rax
  int v15; // eax
  unsigned __int64 v16; // rsi
  __int64 *v17; // rdi
  __int64 v18; // rax
  unsigned int v20; // [rsp+48h] [rbp-228h] BYREF
  char v21; // [rsp+4Ch] [rbp-224h]
  __int64 v22; // [rsp+50h] [rbp-220h] BYREF
  __int16 v23; // [rsp+58h] [rbp-218h]
  __int64 v24; // [rsp+60h] [rbp-210h] BYREF
  __int16 v25; // [rsp+68h] [rbp-208h]
  __int64 v26; // [rsp+70h] [rbp-200h] BYREF
  __int16 v27; // [rsp+78h] [rbp-1F8h]
  __int64 v28; // [rsp+80h] [rbp-1F0h] BYREF
  __int16 v29; // [rsp+88h] [rbp-1E8h]
  __int64 v30; // [rsp+90h] [rbp-1E0h] BYREF
  __int16 v31; // [rsp+98h] [rbp-1D8h]
  __int64 v32; // [rsp+A0h] [rbp-1D0h] BYREF
  __int16 v33; // [rsp+A8h] [rbp-1C8h]
  __int64 v34; // [rsp+B0h] [rbp-1C0h] BYREF
  __int16 v35; // [rsp+B8h] [rbp-1B8h]
  __int64 v36; // [rsp+C0h] [rbp-1B0h] BYREF
  __int16 v37; // [rsp+C8h] [rbp-1A8h]
  unsigned __int64 v38; // [rsp+D0h] [rbp-1A0h] BYREF
  __int16 v39; // [rsp+D8h] [rbp-198h]
  __int64 v40; // [rsp+E0h] [rbp-190h] BYREF
  __int64 v41; // [rsp+E8h] [rbp-188h]
  __int64 v42; // [rsp+F0h] [rbp-180h]
  _QWORD v43[4]; // [rsp+100h] [rbp-170h] BYREF
  __int64 v44[4]; // [rsp+120h] [rbp-150h] BYREF
  _QWORD v45[4]; // [rsp+140h] [rbp-130h] BYREF
  __int64 v46[4]; // [rsp+160h] [rbp-110h] BYREF
  _QWORD v47[4]; // [rsp+180h] [rbp-F0h] BYREF
  __m128i v48; // [rsp+1A0h] [rbp-D0h] BYREF
  char v49; // [rsp+1B0h] [rbp-C0h]
  char v50; // [rsp+1B1h] [rbp-BFh]
  __m128i v51; // [rsp+1C0h] [rbp-B0h] BYREF
  __int16 v52; // [rsp+1D0h] [rbp-A0h]
  __m128i v53[2]; // [rsp+1E0h] [rbp-90h] BYREF
  __m128i v54; // [rsp+200h] [rbp-70h] BYREF
  char v55; // [rsp+210h] [rbp-60h]
  char v56; // [rsp+211h] [rbp-5Fh]
  __m128i v57; // [rsp+220h] [rbp-50h] BYREF
  char v58; // [rsp+230h] [rbp-40h]
  char v59; // [rsp+231h] [rbp-3Fh]

  v23 = 256;
  v25 = 256;
  v29 = 256;
  v43[2] = 0xFFFFFFFFLL;
  v27 = 256;
  v45[2] = 0xFFFFFFFFLL;
  v40 = 0;
  v41 = 0;
  v42 = 0xFFFF;
  v22 = 0;
  v24 = 0;
  v43[0] = 0;
  v43[1] = 0;
  v26 = 0;
  v28 = 0;
  v44[0] = 0;
  v44[1] = 0;
  v44[2] = -1;
  v45[0] = 0;
  v45[1] = 0;
  v46[0] = 0;
  v46[1] = 0;
  v46[2] = -1;
  v20 = 0;
  v21 = 0;
  v39 = 256;
  v7 = a1 + 8;
  v30 = 0;
  v31 = 256;
  v47[0] = 0;
  v47[1] = 0;
  v47[2] = 0xFFFF;
  v32 = 0;
  v33 = 256;
  v34 = 0;
  v35 = 256;
  v36 = 0;
  v37 = 256;
  v38 = 0;
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
                              if ( sub_2241AC0(a1 + 72, "runtimeLang") )
                              {
                                if ( sub_2241AC0(a1 + 72, "vtableHolder") )
                                {
                                  if ( sub_2241AC0(a1 + 72, "templateParams") )
                                  {
                                    if ( sub_2241AC0(a1 + 72, "identifier") )
                                    {
                                      if ( sub_2241AC0(a1 + 72, "discriminator") )
                                      {
                                        v56 = 1;
                                        v54.m128i_i64[0] = (__int64)"'";
                                        v48.m128i_i64[0] = (__int64)"invalid field '";
                                        v55 = 3;
                                        v52 = 260;
                                        v51.m128i_i64[0] = a1 + 72;
                                        v50 = 1;
                                        v49 = 3;
                                        sub_14EC200(v53, &v48, &v51);
                                        sub_14EC200(&v57, v53, &v54);
                                        v9 = sub_38814C0(v7, *(_QWORD *)(a1 + 56), (__int64)&v57);
                                      }
                                      else
                                      {
                                        v9 = sub_38A29E0(a1, (__int64)"discriminator", 13, (__int64)&v38, a4, a5, a6);
                                      }
                                    }
                                    else
                                    {
                                      v9 = sub_388B8F0(a1, (__int64)"identifier", 10, (__int64)&v36);
                                    }
                                  }
                                  else
                                  {
                                    v9 = sub_38A29E0(a1, (__int64)"templateParams", 14, (__int64)&v34, a4, a5, a6);
                                  }
                                }
                                else
                                {
                                  v9 = sub_38A29E0(a1, (__int64)"vtableHolder", 12, (__int64)&v32, a4, a5, a6);
                                }
                              }
                              else
                              {
                                v9 = sub_3889810(a1, (__int64)"runtimeLang", 11, (__int64)v47);
                              }
                            }
                            else
                            {
                              v9 = sub_38A29E0(a1, (__int64)"elements", 8, (__int64)&v30, a4, a5, a6);
                            }
                          }
                          else
                          {
                            v9 = sub_388BBA0(a1, (__int64)"flags", 5, (__int64)&v20);
                          }
                        }
                        else
                        {
                          v9 = sub_3889510(a1, (__int64)"offset", 6, (__int64)v46);
                        }
                      }
                      else
                      {
                        v9 = sub_3889510(a1, (__int64)"align", 5, (__int64)v45);
                      }
                    }
                    else
                    {
                      v9 = sub_3889510(a1, (__int64)"size", 4, (__int64)v44);
                    }
                  }
                  else
                  {
                    v9 = sub_38A29E0(a1, (__int64)"baseType", 8, (__int64)&v28, a4, a5, a6);
                  }
                }
                else
                {
                  v9 = sub_38A29E0(a1, (__int64)"scope", 5, (__int64)&v26, a4, a5, a6);
                }
              }
              else
              {
                v9 = sub_38895C0(a1, (__int64)"line", 4, (__int64)v43);
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
          v9 = sub_38899B0(a1, (__int64)"tag", 3, (__int64)&v40);
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
    v59 = 1;
    v58 = 3;
    v57.m128i_i64[0] = (__int64)"expected field label here";
    if ( (unsigned __int8)sub_38814C0(v7, v16, (__int64)&v57) )
      return 1;
  }
LABEL_8:
  v10 = *(_QWORD *)(a1 + 56);
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v11 )
  {
    if ( !(_BYTE)v41 )
    {
      v59 = 1;
      v58 = 3;
      v57.m128i_i64[0] = (__int64)"missing required field 'tag'";
      return (unsigned int)sub_38814C0(v7, v10, (__int64)&v57);
    }
    v12 = v36;
    if ( v36 )
    {
      v13 = sub_15BE040(
              *(__int64 **)a1,
              v36,
              v40,
              v22,
              v24,
              v43[0],
              v26,
              v28,
              v44[0],
              v45[0],
              v46[0],
              v20,
              v30,
              v47[0],
              v32,
              v34,
              v38);
      if ( v13 )
      {
        *a2 = v13;
        return v11;
      }
      v12 = v36;
    }
    v17 = *(__int64 **)a1;
    if ( a3 )
      v18 = sub_15BDB40(
              v17,
              v40,
              v22,
              v24,
              v43[0],
              v26,
              v28,
              v44[0],
              v45[0],
              v46[0],
              v20,
              v30,
              v47[0],
              v32,
              v34,
              v12,
              v38,
              1u,
              1);
    else
      v18 = sub_15BDB40(
              v17,
              v40,
              v22,
              v24,
              v43[0],
              v26,
              v28,
              v44[0],
              v45[0],
              v46[0],
              v20,
              v30,
              v47[0],
              v32,
              v34,
              v12,
              v38,
              0,
              1);
    *a2 = v18;
  }
  return v11;
}

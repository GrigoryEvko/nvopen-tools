// Function: sub_122A6C0
// Address: 0x122a6c0
//
__int64 __fastcall sub_122A6C0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v4; // r13
  int v5; // eax
  char v6; // al
  unsigned int v7; // r13d
  _QWORD *v8; // rdi
  __int64 v9; // rax
  int v11; // eax
  unsigned __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int16 v19; // [rsp+1Ch] [rbp-1E4h] BYREF
  __int16 v20; // [rsp+1Eh] [rbp-1E2h] BYREF
  __int64 v21; // [rsp+20h] [rbp-1E0h] BYREF
  __int16 v22; // [rsp+28h] [rbp-1D8h]
  __int64 v23; // [rsp+30h] [rbp-1D0h] BYREF
  __int16 v24; // [rsp+38h] [rbp-1C8h]
  __int64 v25; // [rsp+40h] [rbp-1C0h] BYREF
  __int16 v26; // [rsp+48h] [rbp-1B8h]
  __int64 v27; // [rsp+50h] [rbp-1B0h] BYREF
  __int16 v28; // [rsp+58h] [rbp-1A8h]
  __int64 v29; // [rsp+60h] [rbp-1A0h] BYREF
  __int16 v30; // [rsp+68h] [rbp-198h]
  __int64 v31; // [rsp+70h] [rbp-190h] BYREF
  __int16 v32; // [rsp+78h] [rbp-188h]
  __int64 v33; // [rsp+80h] [rbp-180h] BYREF
  __int16 v34; // [rsp+88h] [rbp-178h]
  __int64 v35; // [rsp+90h] [rbp-170h] BYREF
  __int16 v36; // [rsp+98h] [rbp-168h]
  _QWORD v37[4]; // [rsp+A0h] [rbp-160h] BYREF
  _QWORD v38[4]; // [rsp+C0h] [rbp-140h] BYREF
  __m128i v39[2]; // [rsp+E0h] [rbp-120h] BYREF
  char v40; // [rsp+100h] [rbp-100h]
  char v41; // [rsp+101h] [rbp-FFh]
  __m128i v42[2]; // [rsp+110h] [rbp-F0h] BYREF
  __int16 v43; // [rsp+130h] [rbp-D0h]
  __m128i v44[3]; // [rsp+140h] [rbp-C0h] BYREF
  __m128i v45[2]; // [rsp+170h] [rbp-90h] BYREF
  char v46; // [rsp+190h] [rbp-70h]
  char v47; // [rsp+191h] [rbp-6Fh]
  __m128i v48[2]; // [rsp+1A0h] [rbp-60h] BYREF
  char v49; // [rsp+1C0h] [rbp-40h]
  char v50; // [rsp+1C1h] [rbp-3Fh]

  v34 = 256;
  v4 = a1 + 176;
  v22 = 0;
  v24 = 256;
  v28 = 256;
  v21 = 0;
  v23 = 0;
  v25 = 0;
  v26 = 256;
  v27 = 0;
  v37[0] = 0;
  v37[1] = 0;
  v37[2] = 0xFFFFFFFFLL;
  v29 = 0;
  v30 = 256;
  v19 = 0;
  v20 = 1;
  v31 = 0;
  v32 = 256;
  v33 = 0;
  v38[0] = 0;
  v38[1] = 0;
  v38[2] = 0xFFFFFFFFLL;
  v35 = 0;
  v36 = 256;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  else
  {
    v5 = *(_DWORD *)(a1 + 240);
    if ( v5 != 13 )
    {
      if ( v5 != 507 )
      {
LABEL_15:
        v50 = 1;
        v12 = *(_QWORD *)(a1 + 232);
        v49 = 3;
        v48[0].m128i_i64[0] = (__int64)"expected field label here";
        sub_11FD800(v4, v12, (__int64)v48, 1);
        return 1;
      }
      while ( 1 )
      {
        if ( (unsigned int)sub_2241AC0(a1 + 248, "name") )
        {
          if ( (unsigned int)sub_2241AC0(a1 + 248, "scope") )
          {
            if ( (unsigned int)sub_2241AC0(a1 + 248, "linkageName") )
            {
              if ( (unsigned int)sub_2241AC0(a1 + 248, "file") )
              {
                if ( (unsigned int)sub_2241AC0(a1 + 248, "line") )
                {
                  if ( (unsigned int)sub_2241AC0(a1 + 248, "type") )
                  {
                    if ( (unsigned int)sub_2241AC0(a1 + 248, "isLocal") )
                    {
                      if ( (unsigned int)sub_2241AC0(a1 + 248, "isDefinition") )
                      {
                        if ( (unsigned int)sub_2241AC0(a1 + 248, "templateParams") )
                        {
                          if ( (unsigned int)sub_2241AC0(a1 + 248, "declaration") )
                          {
                            if ( (unsigned int)sub_2241AC0(a1 + 248, "align") )
                            {
                              if ( (unsigned int)sub_2241AC0(a1 + 248, "annotations") )
                              {
                                v47 = 1;
                                v42[0].m128i_i64[0] = a1 + 248;
                                v45[0].m128i_i64[0] = (__int64)"'";
                                v39[0].m128i_i64[0] = (__int64)"invalid field '";
                                v46 = 3;
                                v43 = 260;
                                v41 = 1;
                                v40 = 3;
                                sub_9C6370(v44, v39, v42, v13, v14, v15);
                                sub_9C6370(v48, v44, v45, v16, v17, v18);
                                sub_11FD800(v4, *(_QWORD *)(a1 + 232), (__int64)v48, 1);
                                return 1;
                              }
                              v6 = sub_1225DC0(a1, (__int64)"annotations", 11, (__int64)&v35);
                            }
                            else
                            {
                              v6 = sub_1208450(a1, (__int64)"align", 5, (__int64)v38);
                            }
                          }
                          else
                          {
                            v6 = sub_1225DC0(a1, (__int64)"declaration", 11, (__int64)&v33);
                          }
                        }
                        else
                        {
                          v6 = sub_1225DC0(a1, (__int64)"templateParams", 14, (__int64)&v31);
                        }
                      }
                      else
                      {
                        v6 = sub_1207D20(a1, (__int64)"isDefinition", 12, (__int64)&v20);
                      }
                    }
                    else
                    {
                      v6 = sub_1207D20(a1, (__int64)"isLocal", 7, (__int64)&v19);
                    }
                  }
                  else
                  {
                    v6 = sub_1225DC0(a1, (__int64)"type", 4, (__int64)&v29);
                  }
                }
                else
                {
                  v6 = sub_1208380(a1, (__int64)"line", 4, (__int64)v37);
                }
              }
              else
              {
                v6 = sub_1225DC0(a1, (__int64)"file", 4, (__int64)&v27);
              }
            }
            else
            {
              v6 = sub_120BB20(a1, "linkageName", 11, (__int64)&v25);
            }
          }
          else
          {
            v6 = sub_1225DC0(a1, (__int64)"scope", 5, (__int64)&v23);
          }
        }
        else
        {
          v6 = sub_120BB20(a1, "name", 4, (__int64)&v21);
        }
        if ( v6 )
          return 1;
        if ( *(_DWORD *)(a1 + 240) != 4 )
          break;
        v11 = sub_1205200(v4);
        *(_DWORD *)(a1 + 240) = v11;
        if ( v11 != 507 )
          goto LABEL_15;
      }
    }
    v7 = sub_120AFE0(a1, 13, "expected ')' here");
    if ( (_BYTE)v7 )
      return 1;
    v8 = *(_QWORD **)a1;
    if ( a3 )
      v9 = sub_B0B820(v8, v23, v21, v25, v27, v37[0], v29, v19, v20, v33, v31, v38[0], v35, 1u, 1);
    else
      v9 = sub_B0B820(v8, v23, v21, v25, v27, v37[0], v29, v19, v20, v33, v31, v38[0], v35, 0, 1);
    *a2 = v9;
  }
  return v7;
}

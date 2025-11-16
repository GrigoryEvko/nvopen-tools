// Function: sub_1226670
// Address: 0x1226670
//
__int64 __fastcall sub_1226670(__int64 a1, __int64 *a2, char a3)
{
  __int64 v4; // r13
  int v5; // eax
  char v6; // al
  unsigned __int64 v7; // r15
  unsigned int v8; // r14d
  __int32 v9; // edx
  int v10; // eax
  __int8 v11; // cl
  const char *v12; // rax
  int v14; // eax
  unsigned __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int128 v22; // [rsp-28h] [rbp-2E8h]
  __int16 v23; // [rsp+32h] [rbp-28Eh] BYREF
  __int16 v24; // [rsp+34h] [rbp-28Ch] BYREF
  __int16 v25; // [rsp+36h] [rbp-28Ah] BYREF
  int v26; // [rsp+38h] [rbp-288h] BYREF
  char v27; // [rsp+3Ch] [rbp-284h]
  __int64 v28; // [rsp+40h] [rbp-280h] BYREF
  __int16 v29; // [rsp+48h] [rbp-278h]
  __int64 v30; // [rsp+50h] [rbp-270h] BYREF
  __int16 v31; // [rsp+58h] [rbp-268h]
  _BYTE *v32; // [rsp+60h] [rbp-260h] BYREF
  __int16 v33; // [rsp+68h] [rbp-258h]
  __int64 v34; // [rsp+70h] [rbp-250h] BYREF
  __int16 v35; // [rsp+78h] [rbp-248h]
  __int64 v36; // [rsp+80h] [rbp-240h] BYREF
  __int16 v37; // [rsp+88h] [rbp-238h]
  __int64 v38; // [rsp+90h] [rbp-230h] BYREF
  __int16 v39; // [rsp+98h] [rbp-228h]
  __int64 v40; // [rsp+A0h] [rbp-220h] BYREF
  __int64 v41; // [rsp+A8h] [rbp-218h]
  __int64 v42; // [rsp+B0h] [rbp-210h]
  _QWORD v43[4]; // [rsp+C0h] [rbp-200h] BYREF
  __int64 v44[4]; // [rsp+E0h] [rbp-1E0h] BYREF
  _QWORD v45[4]; // [rsp+100h] [rbp-1C0h] BYREF
  __int64 v46[4]; // [rsp+120h] [rbp-1A0h] BYREF
  _QWORD v47[4]; // [rsp+140h] [rbp-180h] BYREF
  _QWORD v48[4]; // [rsp+160h] [rbp-160h] BYREF
  _QWORD v49[4]; // [rsp+180h] [rbp-140h] BYREF
  __m128i v50[2]; // [rsp+1A0h] [rbp-120h] BYREF
  char v51; // [rsp+1C0h] [rbp-100h]
  char v52; // [rsp+1C1h] [rbp-FFh]
  __m128i v53; // [rsp+1D0h] [rbp-F0h] BYREF
  __int16 v54; // [rsp+1F0h] [rbp-D0h]
  __m128i v55; // [rsp+200h] [rbp-C0h] BYREF
  __m128i v56[2]; // [rsp+230h] [rbp-90h] BYREF
  char v57; // [rsp+250h] [rbp-70h]
  char v58; // [rsp+251h] [rbp-6Fh]
  __m128i v59[2]; // [rsp+260h] [rbp-60h] BYREF
  char v60; // [rsp+280h] [rbp-40h]
  char v61; // [rsp+281h] [rbp-3Fh]

  v29 = 256;
  v31 = 256;
  v35 = 256;
  v43[2] = 0xFFFFFFFFLL;
  v33 = 256;
  v45[2] = 0xFFFFFFFFLL;
  v40 = 0;
  v41 = 0;
  v42 = 0xFFFF;
  v28 = 0;
  v30 = 0;
  v43[0] = 0;
  v43[1] = 0;
  v32 = 0;
  v34 = 0;
  v44[0] = 0;
  v44[1] = 0;
  v44[2] = -1;
  v45[0] = 0;
  v45[1] = 0;
  v46[0] = 0;
  v46[1] = 0;
  v46[2] = -1;
  v26 = 0;
  v27 = 0;
  v25 = 0;
  v4 = a1 + 176;
  v37 = 256;
  v36 = 0;
  v47[0] = 0xFFFFFFFFLL;
  v47[1] = 0;
  v47[2] = 0xFFFFFFFFLL;
  v38 = 0;
  v39 = 256;
  v48[0] = 0;
  v48[1] = 0;
  v48[2] = 7;
  v23 = 0;
  v49[0] = 0;
  v49[1] = 0;
  v49[2] = 0xFFFF;
  v24 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    return 1;
  v5 = *(_DWORD *)(a1 + 240);
  if ( v5 != 13 )
  {
    if ( v5 != 507 )
    {
LABEL_23:
      v61 = 1;
      v15 = *(_QWORD *)(a1 + 232);
      v60 = 3;
      v59[0].m128i_i64[0] = (__int64)"expected field label here";
      sub_11FD800(v4, v15, (__int64)v59, 1);
      return 1;
    }
    while ( 1 )
    {
      if ( (unsigned int)sub_2241AC0(a1 + 248, "tag") )
      {
        if ( (unsigned int)sub_2241AC0(a1 + 248, "name") )
        {
          if ( (unsigned int)sub_2241AC0(a1 + 248, "file") )
          {
            if ( (unsigned int)sub_2241AC0(a1 + 248, "line") )
            {
              if ( (unsigned int)sub_2241AC0(a1 + 248, "scope") )
              {
                if ( (unsigned int)sub_2241AC0(a1 + 248, "baseType") )
                {
                  if ( (unsigned int)sub_2241AC0(a1 + 248, "size") )
                  {
                    if ( (unsigned int)sub_2241AC0(a1 + 248, "align") )
                    {
                      if ( (unsigned int)sub_2241AC0(a1 + 248, "offset") )
                      {
                        if ( (unsigned int)sub_2241AC0(a1 + 248, "flags") )
                        {
                          if ( (unsigned int)sub_2241AC0(a1 + 248, "extraData") )
                          {
                            if ( (unsigned int)sub_2241AC0(a1 + 248, "dwarfAddressSpace") )
                            {
                              if ( (unsigned int)sub_2241AC0(a1 + 248, "annotations") )
                              {
                                if ( (unsigned int)sub_2241AC0(a1 + 248, "ptrAuthKey") )
                                {
                                  if ( (unsigned int)sub_2241AC0(a1 + 248, "ptrAuthIsAddressDiscriminated") )
                                  {
                                    if ( (unsigned int)sub_2241AC0(a1 + 248, "ptrAuthExtraDiscriminator") )
                                    {
                                      if ( (unsigned int)sub_2241AC0(a1 + 248, "ptrAuthIsaPointer") )
                                      {
                                        if ( (unsigned int)sub_2241AC0(a1 + 248, "ptrAuthAuthenticatesNullValues") )
                                        {
                                          v58 = 1;
                                          v53.m128i_i64[0] = a1 + 248;
                                          v56[0].m128i_i64[0] = (__int64)"'";
                                          v50[0].m128i_i64[0] = (__int64)"invalid field '";
                                          v57 = 3;
                                          v54 = 260;
                                          v52 = 1;
                                          v51 = 3;
                                          sub_9C6370(&v55, v50, &v53, v16, v17, v18);
                                          sub_9C6370(v59, &v55, v56, v19, v20, v21);
                                          sub_11FD800(v4, *(_QWORD *)(a1 + 232), (__int64)v59, 1);
                                          return 1;
                                        }
                                        v6 = sub_1207D20(
                                               a1,
                                               (__int64)"ptrAuthAuthenticatesNullValues",
                                               30,
                                               (__int64)&v25);
                                      }
                                      else
                                      {
                                        v6 = sub_1207D20(a1, (__int64)"ptrAuthIsaPointer", 17, (__int64)&v24);
                                      }
                                    }
                                    else
                                    {
                                      v6 = sub_1208450(a1, (__int64)"ptrAuthExtraDiscriminator", 25, (__int64)v49);
                                    }
                                  }
                                  else
                                  {
                                    v6 = sub_1207D20(a1, (__int64)"ptrAuthIsAddressDiscriminated", 29, (__int64)&v23);
                                  }
                                }
                                else
                                {
                                  v6 = sub_1208450(a1, (__int64)"ptrAuthKey", 10, (__int64)v48);
                                }
                              }
                              else
                              {
                                v6 = sub_1225DC0(a1, (__int64)"annotations", 11, (__int64)&v38);
                              }
                            }
                            else
                            {
                              v6 = sub_1208450(a1, (__int64)"dwarfAddressSpace", 17, (__int64)v47);
                            }
                          }
                          else
                          {
                            v6 = sub_1225DC0(a1, (__int64)"extraData", 9, (__int64)&v36);
                          }
                        }
                        else
                        {
                          v6 = sub_120BE50(a1, (__int64)"flags", 5, (__int64)&v26);
                        }
                      }
                      else
                      {
                        v6 = sub_1208450(a1, (__int64)"offset", 6, (__int64)v46);
                      }
                    }
                    else
                    {
                      v6 = sub_1208450(a1, (__int64)"align", 5, (__int64)v45);
                    }
                  }
                  else
                  {
                    v6 = sub_1208450(a1, (__int64)"size", 4, (__int64)v44);
                  }
                }
                else
                {
                  v6 = sub_1225DC0(a1, (__int64)"baseType", 8, (__int64)&v34);
                }
              }
              else
              {
                v6 = sub_1225DC0(a1, (__int64)"scope", 5, (__int64)&v32);
              }
            }
            else
            {
              v6 = sub_1208380(a1, (__int64)"line", 4, (__int64)v43);
            }
          }
          else
          {
            v6 = sub_1225DC0(a1, (__int64)"file", 4, (__int64)&v30);
          }
        }
        else
        {
          v6 = sub_120BB20(a1, "name", 4, (__int64)&v28);
        }
      }
      else
      {
        v6 = sub_1208B00(a1, (__int64)"tag", 3, (__int64)&v40);
      }
      if ( v6 )
        return 1;
      if ( *(_DWORD *)(a1 + 240) != 4 )
        break;
      v14 = sub_1205200(v4);
      *(_DWORD *)(a1 + 240) = v14;
      if ( v14 != 507 )
        goto LABEL_23;
    }
  }
  v7 = *(_QWORD *)(a1 + 232);
  v8 = sub_120AFE0(a1, 13, "expected ')' here");
  if ( (_BYTE)v8 )
    return 1;
  if ( !(_BYTE)v41 )
  {
    v61 = 1;
    v12 = "missing required field 'tag'";
LABEL_16:
    v59[0].m128i_i64[0] = (__int64)v12;
    v60 = 3;
    sub_11FD800(v4, v7, (__int64)v59, 1);
    return 1;
  }
  if ( !(_BYTE)v35 )
  {
    v61 = 1;
    v12 = "missing required field 'baseType'";
    goto LABEL_16;
  }
  v9 = v47[0];
  v53.m128i_i64[0] = 0;
  v55.m128i_i64[0] = 0;
  v10 = 0;
  if ( v47[0] == 0xFFFFFFFFLL )
    v9 = 0;
  v11 = 0;
  if ( v48[0] )
  {
    v10 = (16 * ((_BYTE)v23 != 0)) | (32 * LODWORD(v49[0])) | LODWORD(v48[0]);
    if ( (_BYTE)v24 )
      v10 |= 0x200000u;
    if ( (_BYTE)v25 )
      v10 |= 0x400000u;
    v11 = 1;
  }
  v53.m128i_i32[0] = v9;
  v55.m128i_i32[0] = v10;
  v53.m128i_i8[4] = v47[0] != 0xFFFFFFFFLL;
  v55.m128i_i8[4] = v11;
  v56[0].m128i_i64[0] = v53.m128i_i64[0];
  v59[0].m128i_i64[0] = v55.m128i_i64[0];
  *((_QWORD *)&v22 + 1) = v38;
  *(_QWORD *)&v22 = v36;
  *a2 = sub_B05AE0(
          *(_QWORD **)a1,
          v40,
          v28,
          v30,
          v43[0],
          v32,
          v34,
          v44[0],
          v45[0],
          v46[0],
          v53.m128i_i64[0],
          v55.m128i_i64[0],
          v26,
          v22,
          a3 != 0,
          1);
  return v8;
}

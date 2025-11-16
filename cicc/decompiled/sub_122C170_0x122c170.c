// Function: sub_122C170
// Address: 0x122c170
//
__int64 __fastcall sub_122C170(_QWORD **a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v5; // eax
  char v6; // al
  unsigned int v7; // r13d
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // r15
  _QWORD *v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rax
  int v14; // eax
  unsigned __int64 v15; // rsi
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  _QWORD *v35; // [rsp+8h] [rbp-2E8h]
  _QWORD *v36; // [rsp+10h] [rbp-2E0h]
  unsigned int v37; // [rsp+28h] [rbp-2C8h] BYREF
  char v38; // [rsp+2Ch] [rbp-2C4h]
  __int64 v39; // [rsp+30h] [rbp-2C0h] BYREF
  __int16 v40; // [rsp+38h] [rbp-2B8h]
  __int64 v41; // [rsp+40h] [rbp-2B0h] BYREF
  __int16 v42; // [rsp+48h] [rbp-2A8h]
  __int64 v43; // [rsp+50h] [rbp-2A0h] BYREF
  __int16 v44; // [rsp+58h] [rbp-298h]
  __int64 v45; // [rsp+60h] [rbp-290h] BYREF
  __int16 v46; // [rsp+68h] [rbp-288h]
  _QWORD v47[4]; // [rsp+70h] [rbp-280h] BYREF
  __int64 v48[4]; // [rsp+90h] [rbp-260h] BYREF
  _QWORD v49[4]; // [rsp+B0h] [rbp-240h] BYREF
  __m128i v50[2]; // [rsp+D0h] [rbp-220h] BYREF
  char v51; // [rsp+F0h] [rbp-200h]
  char v52; // [rsp+F1h] [rbp-1FFh]
  __m128i v53[2]; // [rsp+100h] [rbp-1F0h] BYREF
  __int16 v54; // [rsp+120h] [rbp-1D0h]
  __m128i v55[3]; // [rsp+130h] [rbp-1C0h] BYREF
  __m128i v56[2]; // [rsp+160h] [rbp-190h] BYREF
  char v57; // [rsp+180h] [rbp-170h]
  char v58; // [rsp+181h] [rbp-16Fh]
  __m128i v59[2]; // [rsp+190h] [rbp-160h] BYREF
  char v60; // [rsp+1B0h] [rbp-140h]
  char v61; // [rsp+1B1h] [rbp-13Fh]
  __m128i v62; // [rsp+1C0h] [rbp-130h] BYREF
  unsigned __int64 v63; // [rsp+1D0h] [rbp-120h]
  __int64 v64; // [rsp+1D8h] [rbp-118h]
  _QWORD *v65; // [rsp+1E0h] [rbp-110h]
  __int64 v66; // [rsp+1E8h] [rbp-108h]
  __int64 v67; // [rsp+1F0h] [rbp-100h]
  __m128i v68; // [rsp+200h] [rbp-F0h] BYREF
  unsigned __int64 v69; // [rsp+210h] [rbp-E0h]
  __int64 v70; // [rsp+218h] [rbp-D8h]
  _QWORD *v71; // [rsp+220h] [rbp-D0h]
  __int64 v72; // [rsp+228h] [rbp-C8h]
  __int64 v73; // [rsp+230h] [rbp-C0h]
  __m128i v74; // [rsp+240h] [rbp-B0h] BYREF
  unsigned __int64 v75; // [rsp+250h] [rbp-A0h]
  __int64 v76; // [rsp+258h] [rbp-98h]
  _QWORD *v77; // [rsp+260h] [rbp-90h]
  __int64 v78; // [rsp+268h] [rbp-88h]
  __int64 v79; // [rsp+270h] [rbp-80h]
  __m128i v80; // [rsp+280h] [rbp-70h] BYREF
  unsigned __int64 v81; // [rsp+290h] [rbp-60h]
  __int64 v82; // [rsp+298h] [rbp-58h]
  _QWORD *v83; // [rsp+2A0h] [rbp-50h]
  __int64 v84; // [rsp+2A8h] [rbp-48h]
  __int64 v85; // [rsp+2B0h] [rbp-40h]

  v3 = (__int64)(a1 + 22);
  v40 = 256;
  v42 = 256;
  v47[2] = 0xFFFFFFFFLL;
  v49[2] = 0xFFFFFFFFLL;
  v46 = 256;
  v63 = 0x8000000000000000LL;
  v44 = 256;
  v64 = 0x7FFFFFFFFFFFFFFFLL;
  v39 = 0;
  v41 = 0;
  v47[0] = 0;
  v47[1] = 0;
  v43 = 0;
  v45 = 0;
  v48[0] = 0;
  v48[1] = 0;
  v48[2] = -1;
  v49[0] = 0;
  v49[1] = 0;
  v37 = 0;
  v38 = 0;
  v62 = 0u;
  v65 = 0;
  v66 = 256;
  v69 = 0x8000000000000000LL;
  v75 = 0x8000000000000000LL;
  v81 = 0x8000000000000000LL;
  v67 = 0;
  v68 = 0u;
  v70 = 0x7FFFFFFFFFFFFFFFLL;
  v71 = 0;
  v72 = 256;
  v73 = 0;
  v74 = 0u;
  v76 = 0x7FFFFFFFFFFFFFFFLL;
  v77 = 0;
  v78 = 256;
  v79 = 0;
  v80 = 0u;
  v82 = 0x7FFFFFFFFFFFFFFFLL;
  v83 = 0;
  v84 = 256;
  v85 = 0;
  *((_DWORD *)a1 + 60) = sub_1205200((__int64)(a1 + 22));
  if ( (unsigned __int8)sub_120AFE0((__int64)a1, 12, "expected '(' here") )
  {
    return 1;
  }
  else
  {
    v5 = *((_DWORD *)a1 + 60);
    if ( v5 != 13 )
    {
      if ( v5 != 507 )
      {
LABEL_27:
        v15 = (unsigned __int64)a1[29];
        v61 = 1;
        v59[0].m128i_i64[0] = (__int64)"expected field label here";
        v60 = 3;
        sub_11FD800(v3, v15, (__int64)v59, 1);
        return 1;
      }
      while ( 1 )
      {
        if ( (unsigned int)sub_2241AC0(a1 + 31, "name") )
        {
          if ( (unsigned int)sub_2241AC0(a1 + 31, "file") )
          {
            if ( (unsigned int)sub_2241AC0(a1 + 31, "line") )
            {
              if ( (unsigned int)sub_2241AC0(a1 + 31, "scope") )
              {
                if ( (unsigned int)sub_2241AC0(a1 + 31, "baseType") )
                {
                  if ( (unsigned int)sub_2241AC0(a1 + 31, "size") )
                  {
                    if ( (unsigned int)sub_2241AC0(a1 + 31, "align") )
                    {
                      if ( (unsigned int)sub_2241AC0(a1 + 31, "flags") )
                      {
                        if ( (unsigned int)sub_2241AC0(a1 + 31, "lowerBound") )
                        {
                          if ( (unsigned int)sub_2241AC0(a1 + 31, "upperBound") )
                          {
                            if ( (unsigned int)sub_2241AC0(a1 + 31, "stride") )
                            {
                              if ( (unsigned int)sub_2241AC0(a1 + 31, "bias") )
                              {
                                v53[0].m128i_i64[0] = (__int64)(a1 + 31);
                                v56[0].m128i_i64[0] = (__int64)"'";
                                v50[0].m128i_i64[0] = (__int64)"invalid field '";
                                v58 = 1;
                                v57 = 3;
                                v54 = 260;
                                v52 = 1;
                                v51 = 3;
                                sub_9C6370(v55, v50, v53, v29, v30, v31);
                                sub_9C6370(v59, v55, v56, v32, v33, v34);
                                sub_11FD800(v3, (unsigned __int64)a1[29], (__int64)v59, 1);
                                return 1;
                              }
                              v6 = sub_122C010((__int64)a1, (__int64)"bias", 4, &v80);
                            }
                            else
                            {
                              v6 = sub_122C010((__int64)a1, (__int64)"stride", 6, &v74);
                            }
                          }
                          else
                          {
                            v6 = sub_122C010((__int64)a1, (__int64)"upperBound", 10, &v68);
                          }
                        }
                        else
                        {
                          v6 = sub_122C010((__int64)a1, (__int64)"lowerBound", 10, &v62);
                        }
                      }
                      else
                      {
                        v6 = sub_120BE50((__int64)a1, (__int64)"flags", 5, (__int64)&v37);
                      }
                    }
                    else
                    {
                      v6 = sub_1208450((__int64)a1, (__int64)"align", 5, (__int64)v49);
                    }
                  }
                  else
                  {
                    v6 = sub_1208450((__int64)a1, (__int64)"size", 4, (__int64)v48);
                  }
                }
                else
                {
                  v6 = sub_1225DC0((__int64)a1, (__int64)"baseType", 8, (__int64)&v45);
                }
              }
              else
              {
                v6 = sub_1225DC0((__int64)a1, (__int64)"scope", 5, (__int64)&v43);
              }
            }
            else
            {
              v6 = sub_1208380((__int64)a1, (__int64)"line", 4, (__int64)v47);
            }
          }
          else
          {
            v6 = sub_1225DC0((__int64)a1, (__int64)"file", 4, (__int64)&v41);
          }
        }
        else
        {
          v6 = sub_120BB20((__int64)a1, "name", 4, (__int64)&v39);
        }
        if ( v6 )
          return 1;
        if ( *((_DWORD *)a1 + 60) != 4 )
          break;
        v14 = sub_1205200(v3);
        *((_DWORD *)a1 + 60) = v14;
        if ( v14 != 507 )
          goto LABEL_27;
      }
    }
    v7 = sub_120AFE0((__int64)a1, 13, "expected ')' here");
    if ( (_BYTE)v7 )
      return 1;
    if ( HIDWORD(v67) == 1 )
    {
      v26 = v62.m128i_i64[0];
      v27 = sub_BCB2E0(*a1);
      v28 = sub_ACD640(v27, v26, 1u);
      v35 = sub_B98A20(v28, v26);
    }
    else
    {
      v8 = 0;
      if ( HIDWORD(v67) == 2 )
        v8 = v65;
      v35 = v8;
    }
    if ( HIDWORD(v73) == 1 )
    {
      v23 = v68.m128i_i64[0];
      v24 = sub_BCB2E0(*a1);
      v25 = sub_ACD640(v24, v23, 1u);
      v36 = sub_B98A20(v25, v23);
    }
    else
    {
      v9 = 0;
      if ( HIDWORD(v73) == 2 )
        v9 = v71;
      v36 = v9;
    }
    if ( HIDWORD(v79) == 1 )
    {
      v20 = v74.m128i_i64[0];
      v21 = sub_BCB2E0(*a1);
      v22 = sub_ACD640(v21, v20, 1u);
      v10 = sub_B98A20(v22, v20);
    }
    else
    {
      v10 = 0;
      if ( HIDWORD(v79) == 2 )
        v10 = v77;
    }
    if ( HIDWORD(v85) == 1 )
    {
      v17 = v80.m128i_i64[0];
      v18 = sub_BCB2E0(*a1);
      v19 = sub_ACD640(v18, v17, 1u);
      v11 = sub_B98A20(v19, v17);
    }
    else
    {
      v11 = 0;
      if ( HIDWORD(v85) == 2 )
        v11 = v83;
    }
    v12 = *a1;
    if ( a3 )
      v13 = sub_B03CA0(
              v12,
              v39,
              v41,
              v47[0],
              v43,
              v48[0],
              v49[0],
              v37,
              v45,
              (__int64)v35,
              (unsigned __int64)v36,
              (__int64)v10,
              (__int64)v11,
              1u,
              1);
    else
      v13 = sub_B03CA0(
              v12,
              v39,
              v41,
              v47[0],
              v43,
              v48[0],
              v49[0],
              v37,
              v45,
              (__int64)v35,
              (unsigned __int64)v36,
              (__int64)v10,
              (__int64)v11,
              0,
              1);
    *a2 = v13;
  }
  return v7;
}

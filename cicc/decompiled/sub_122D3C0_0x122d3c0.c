// Function: sub_122D3C0
// Address: 0x122d3c0
//
__int64 __fastcall sub_122D3C0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v5; // r13
  int v6; // eax
  char v7; // al
  unsigned __int64 v8; // r14
  unsigned int v9; // r15d
  _QWORD *v10; // r14
  __int32 v11; // r13d
  __int64 v12; // rsi
  __int64 *v13; // rdi
  __int64 v14; // rax
  __int64 *v16; // rdi
  __int64 v17; // rax
  int v18; // eax
  const char *v19; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rsi
  int v24; // eax
  unsigned int v25; // eax
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  bool v33; // [rsp+7Fh] [rbp-351h]
  unsigned int v34; // [rsp+88h] [rbp-348h] BYREF
  char v35; // [rsp+8Ch] [rbp-344h]
  __int64 v36; // [rsp+90h] [rbp-340h] BYREF
  __int16 v37; // [rsp+98h] [rbp-338h]
  __int64 v38; // [rsp+A0h] [rbp-330h] BYREF
  __int16 v39; // [rsp+A8h] [rbp-328h]
  __int64 v40; // [rsp+B0h] [rbp-320h] BYREF
  __int16 v41; // [rsp+B8h] [rbp-318h]
  __int64 v42; // [rsp+C0h] [rbp-310h] BYREF
  __int16 v43; // [rsp+C8h] [rbp-308h]
  __int64 v44; // [rsp+D0h] [rbp-300h] BYREF
  __int16 v45; // [rsp+D8h] [rbp-2F8h]
  __int64 v46; // [rsp+E0h] [rbp-2F0h] BYREF
  __int16 v47; // [rsp+E8h] [rbp-2E8h]
  __int64 v48; // [rsp+F0h] [rbp-2E0h] BYREF
  __int16 v49; // [rsp+F8h] [rbp-2D8h]
  __int64 v50; // [rsp+100h] [rbp-2D0h] BYREF
  __int16 v51; // [rsp+108h] [rbp-2C8h]
  unsigned __int64 v52; // [rsp+110h] [rbp-2C0h] BYREF
  __int16 v53; // [rsp+118h] [rbp-2B8h]
  __int64 v54; // [rsp+120h] [rbp-2B0h] BYREF
  __int16 v55; // [rsp+128h] [rbp-2A8h]
  __int64 v56; // [rsp+130h] [rbp-2A0h] BYREF
  __int16 v57; // [rsp+138h] [rbp-298h]
  __int64 v58; // [rsp+140h] [rbp-290h] BYREF
  __int16 v59; // [rsp+148h] [rbp-288h]
  __int64 v60; // [rsp+150h] [rbp-280h] BYREF
  __int16 v61; // [rsp+158h] [rbp-278h]
  __int64 v62; // [rsp+160h] [rbp-270h] BYREF
  __int16 v63; // [rsp+168h] [rbp-268h]
  __int64 v64; // [rsp+170h] [rbp-260h] BYREF
  __int64 v65; // [rsp+178h] [rbp-258h]
  __int64 v66; // [rsp+180h] [rbp-250h]
  _QWORD v67[4]; // [rsp+190h] [rbp-240h] BYREF
  __int64 v68[4]; // [rsp+1B0h] [rbp-220h] BYREF
  _QWORD v69[4]; // [rsp+1D0h] [rbp-200h] BYREF
  __int64 v70[4]; // [rsp+1F0h] [rbp-1E0h] BYREF
  _QWORD v71[4]; // [rsp+210h] [rbp-1C0h] BYREF
  __int64 v72; // [rsp+230h] [rbp-1A0h] BYREF
  __int64 v73; // [rsp+238h] [rbp-198h]
  __int64 v74; // [rsp+240h] [rbp-190h]
  _QWORD v75[4]; // [rsp+250h] [rbp-180h] BYREF
  __m128i v76[2]; // [rsp+270h] [rbp-160h] BYREF
  char v77; // [rsp+290h] [rbp-140h]
  char v78; // [rsp+291h] [rbp-13Fh]
  __m128i v79[2]; // [rsp+2A0h] [rbp-130h] BYREF
  __int16 v80; // [rsp+2C0h] [rbp-110h]
  __m128i v81; // [rsp+2D0h] [rbp-100h] BYREF
  char *v82; // [rsp+2E0h] [rbp-F0h]
  __int16 v83; // [rsp+2F0h] [rbp-E0h]
  __m128i v84; // [rsp+300h] [rbp-D0h] BYREF
  const char *v85; // [rsp+310h] [rbp-C0h]
  __int16 v86; // [rsp+320h] [rbp-B0h]
  __m128i v87; // [rsp+330h] [rbp-A0h] BYREF
  char *v88; // [rsp+340h] [rbp-90h]
  __int64 v89; // [rsp+348h] [rbp-88h]
  __int16 v90; // [rsp+350h] [rbp-80h]
  __m128i v91; // [rsp+360h] [rbp-70h] BYREF
  unsigned __int64 v92; // [rsp+370h] [rbp-60h]
  __int64 v93; // [rsp+378h] [rbp-58h]
  _QWORD *v94; // [rsp+380h] [rbp-50h]
  __int64 v95; // [rsp+388h] [rbp-48h]
  __int64 v96; // [rsp+390h] [rbp-40h]

  v37 = 256;
  v41 = 256;
  v39 = 256;
  v67[2] = 0xFFFFFFFFLL;
  v43 = 256;
  v69[2] = 0xFFFFFFFFLL;
  v64 = 0;
  v65 = 0;
  v66 = 0xFFFF;
  v36 = 0;
  v38 = 0;
  v67[0] = 0;
  v67[1] = 0;
  v40 = 0;
  v42 = 0;
  v68[0] = 0;
  v68[1] = 0;
  v68[2] = -1;
  v69[0] = 0;
  v69[1] = 0;
  v70[0] = 0;
  v70[1] = 0;
  v70[2] = -1;
  v34 = 0;
  v35 = 0;
  v53 = 256;
  v5 = a1 + 176;
  v92 = 0x8000000000000000LL;
  v93 = 0x7FFFFFFFFFFFFFFFLL;
  v59 = 256;
  v45 = 256;
  v72 = 0xFFFFFFFFLL;
  v47 = 256;
  v49 = 256;
  v51 = 256;
  v61 = 256;
  v44 = 0;
  v71[0] = 0;
  v71[1] = 0;
  v71[2] = 0xFFFF;
  v73 = 0;
  v74 = 1;
  v46 = 0;
  v48 = 0;
  v50 = 0;
  v52 = 0;
  v54 = 0;
  v55 = 256;
  v56 = 0;
  v57 = 256;
  v58 = 0;
  v91 = 0u;
  v94 = 0;
  v95 = 256;
  v96 = 0;
  v60 = 0;
  v75[0] = 0;
  v75[1] = 0;
  v63 = 256;
  v75[2] = 0xFFFFFFFFLL;
  v62 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    return 1;
  v6 = *(_DWORD *)(a1 + 240);
  if ( v6 != 13 )
  {
    if ( v6 == 507 )
    {
      while ( 1 )
      {
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "tag") )
        {
          v7 = sub_1208B00(a1, (__int64)"tag", 3, (__int64)&v64);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "name") )
        {
          v7 = sub_120BB20(a1, "name", 4, (__int64)&v36);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "file") )
        {
          v7 = sub_1225DC0(a1, (__int64)"file", 4, (__int64)&v38);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "line") )
        {
          v7 = sub_1208380(a1, (__int64)"line", 4, (__int64)v67);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "scope") )
        {
          v7 = sub_1225DC0(a1, (__int64)"scope", 5, (__int64)&v40);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "baseType") )
        {
          v7 = sub_1225DC0(a1, (__int64)"baseType", 8, (__int64)&v42);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "size") )
        {
          v7 = sub_1208450(a1, (__int64)"size", 4, (__int64)v68);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "align") )
        {
          v7 = sub_1208450(a1, (__int64)"align", 5, (__int64)v69);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "offset") )
        {
          v7 = sub_1208450(a1, (__int64)"offset", 6, (__int64)v70);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "flags") )
        {
          v7 = sub_120BE50(a1, (__int64)"flags", 5, (__int64)&v34);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "elements") )
        {
          v7 = sub_1225DC0(a1, (__int64)"elements", 8, (__int64)&v44);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "runtimeLang") )
          break;
        if ( (unsigned int)sub_2241AC0(a1 + 248, "enumKind") )
        {
          if ( (unsigned int)sub_2241AC0(a1 + 248, "vtableHolder") )
          {
            if ( (unsigned int)sub_2241AC0(a1 + 248, "templateParams") )
            {
              if ( (unsigned int)sub_2241AC0(a1 + 248, "identifier") )
              {
                if ( (unsigned int)sub_2241AC0(a1 + 248, "discriminator") )
                {
                  if ( (unsigned int)sub_2241AC0(a1 + 248, "dataLocation") )
                  {
                    if ( (unsigned int)sub_2241AC0(a1 + 248, "associated") )
                    {
                      if ( (unsigned int)sub_2241AC0(a1 + 248, "allocated") )
                      {
                        if ( (unsigned int)sub_2241AC0(a1 + 248, "rank") )
                        {
                          if ( (unsigned int)sub_2241AC0(a1 + 248, "annotations") )
                          {
                            if ( (unsigned int)sub_2241AC0(a1 + 248, "num_extra_inhabitants") )
                            {
                              if ( (unsigned int)sub_2241AC0(a1 + 248, "specification") )
                              {
                                v79[0].m128i_i64[0] = a1 + 248;
                                v84.m128i_i64[0] = (__int64)"'";
                                v76[0].m128i_i64[0] = (__int64)"invalid field '";
                                v86 = 259;
                                v80 = 260;
                                v78 = 1;
                                v77 = 3;
                                sub_9C6370(&v81, v76, v79, v26, v27, v28);
                                sub_9C6370(&v87, &v81, &v84, v29, v30, v31);
                                sub_11FD800(v5, *(_QWORD *)(a1 + 232), (__int64)&v87, 1);
                                return 1;
                              }
                              v7 = sub_1225DC0(a1, (__int64)"specification", 13, (__int64)&v62);
                            }
                            else
                            {
                              v7 = sub_1208450(a1, (__int64)"num_extra_inhabitants", 21, (__int64)v75);
                            }
                          }
                          else
                          {
                            v7 = sub_1225DC0(a1, (__int64)"annotations", 11, (__int64)&v60);
                          }
                        }
                        else
                        {
                          v7 = sub_122C010(a1, (__int64)"rank", 4, &v91);
                        }
                      }
                      else
                      {
                        v7 = sub_1225DC0(a1, (__int64)"allocated", 9, (__int64)&v58);
                      }
                    }
                    else
                    {
                      v7 = sub_1225DC0(a1, (__int64)"associated", 10, (__int64)&v56);
                    }
                  }
                  else
                  {
                    v7 = sub_1225DC0(a1, (__int64)"dataLocation", 12, (__int64)&v54);
                  }
                }
                else
                {
                  v7 = sub_1225DC0(a1, (__int64)"discriminator", 13, (__int64)&v52);
                }
              }
              else
              {
                v7 = sub_120BB20(a1, "identifier", 10, (__int64)&v50);
              }
            }
            else
            {
              v7 = sub_1225DC0(a1, (__int64)"templateParams", 14, (__int64)&v48);
            }
          }
          else
          {
            v7 = sub_1225DC0(a1, (__int64)"vtableHolder", 12, (__int64)&v46);
          }
          goto LABEL_6;
        }
        if ( (_BYTE)v73 )
        {
          v23 = *(_QWORD *)(a1 + 232);
          v87.m128i_i64[0] = (__int64)"field '";
          v88 = "enumKind";
          v84.m128i_i64[0] = (__int64)&v87;
          v90 = 1283;
          v89 = 8;
          v85 = "' cannot be specified more than once";
          v86 = 770;
          sub_11FD800(v5, v23, (__int64)&v84, 1);
          return 1;
        }
        v24 = sub_1205200(v5);
        *(_DWORD *)(a1 + 240) = v24;
        if ( v24 == 529 )
        {
          v7 = sub_1208110(a1, (__int64)"enumKind", 8, (__int64)&v72);
LABEL_6:
          if ( v7 )
            return 1;
          goto LABEL_7;
        }
        if ( v24 != 526 )
        {
          HIBYTE(v90) = 1;
          v19 = "expected DWARF enum kind code";
          goto LABEL_30;
        }
        v25 = sub_E0A660(*(_QWORD *)(a1 + 248), *(_QWORD *)(a1 + 256));
        if ( v25 == -1 )
        {
          v85 = (const char *)(a1 + 248);
          v81.m128i_i64[0] = (__int64)"invalid DWARF enum kind code";
          v82 = " '";
          v84.m128i_i64[0] = (__int64)&v81;
          v87.m128i_i64[0] = (__int64)&v84;
          v83 = 771;
          v86 = 1026;
          v88 = "'";
          v90 = 770;
          goto LABEL_31;
        }
        LOBYTE(v73) = 1;
        v72 = v25;
        *(_DWORD *)(a1 + 240) = sub_1205200(v5);
LABEL_7:
        if ( *(_DWORD *)(a1 + 240) != 4 )
          goto LABEL_8;
        v18 = sub_1205200(v5);
        *(_DWORD *)(a1 + 240) = v18;
        if ( v18 != 507 )
          goto LABEL_29;
      }
      v7 = sub_1208700(a1, (__int64)"runtimeLang", 11, (__int64)v71);
      goto LABEL_6;
    }
LABEL_29:
    HIBYTE(v90) = 1;
    v19 = "expected field label here";
LABEL_30:
    v87.m128i_i64[0] = (__int64)v19;
    LOBYTE(v90) = 3;
LABEL_31:
    sub_11FD800(v5, *(_QWORD *)(a1 + 232), (__int64)&v87, 1);
    return 1;
  }
LABEL_8:
  v8 = *(_QWORD *)(a1 + 232);
  v9 = sub_120AFE0(a1, 13, "expected ')' here");
  if ( (_BYTE)v9 )
    return 1;
  if ( !(_BYTE)v65 )
  {
    v90 = 259;
    v87.m128i_i64[0] = (__int64)"missing required field 'tag'";
    sub_11FD800(v5, v8, (__int64)&v87, 1);
    return 1;
  }
  if ( HIDWORD(v96) == 1 )
  {
    v20 = v91.m128i_i64[0];
    v21 = sub_BCB2E0(*(_QWORD **)a1);
    v22 = sub_ACD640(v21, v20, 1u);
    v10 = sub_B98A20(v22, v20);
  }
  else
  {
    v10 = 0;
    if ( HIDWORD(v96) == 2 )
      v10 = v94;
  }
  v11 = v72;
  v12 = v50;
  v84.m128i_i64[0] = 0;
  if ( v72 == 0xFFFFFFFFLL )
    v11 = 0;
  v33 = v72 != 0xFFFFFFFFLL;
  if ( v50 )
  {
    v84.m128i_i8[4] = v72 != 0xFFFFFFFFLL;
    v13 = *(__int64 **)a1;
    v84.m128i_i32[0] = v11;
    v14 = sub_B06A20(
            v13,
            v50,
            v64,
            v36,
            v38,
            v67[0],
            v40,
            v42,
            v68[0],
            v69[0],
            v70[0],
            v62,
            v75[0],
            v34,
            v44,
            v71[0],
            v84.m128i_i64[0],
            v46,
            v48,
            v52,
            v54,
            v56,
            v58,
            (__int64)v10,
            v60);
    if ( v14 )
    {
      *a2 = v14;
      return v9;
    }
    v12 = v50;
  }
  v84.m128i_i32[0] = v11;
  v16 = *(__int64 **)a1;
  v84.m128i_i8[4] = v33;
  v87.m128i_i64[0] = v84.m128i_i64[0];
  if ( a3 )
    v17 = sub_B065E0(
            v16,
            v64,
            v36,
            v38,
            v67[0],
            v40,
            v42,
            v68[0],
            v69[0],
            v70[0],
            v34,
            v44,
            v71[0],
            v84.m128i_i64[0],
            v46,
            v48,
            v12,
            v52,
            v54,
            v56,
            v58,
            (__int64)v10,
            v60,
            v62,
            v75[0],
            1u,
            1);
  else
    v17 = sub_B065E0(
            v16,
            v64,
            v36,
            v38,
            v67[0],
            v40,
            v42,
            v68[0],
            v69[0],
            v70[0],
            v34,
            v44,
            v71[0],
            v84.m128i_i64[0],
            v46,
            v48,
            v12,
            v52,
            v54,
            v56,
            v58,
            (__int64)v10,
            v60,
            v62,
            v75[0],
            0,
            1);
  *a2 = v17;
  return v9;
}

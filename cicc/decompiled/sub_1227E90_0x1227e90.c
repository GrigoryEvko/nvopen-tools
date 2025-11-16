// Function: sub_1227E90
// Address: 0x1227e90
//
__int64 __fastcall sub_1227E90(__int64 a1, __int64 *a2, char a3)
{
  unsigned __int64 v4; // r14
  __int64 v5; // r12
  unsigned int v6; // eax
  unsigned int v7; // r15d
  int v8; // eax
  char v9; // al
  int v10; // esi
  int v11; // ecx
  int v12; // edx
  int v13; // eax
  int v14; // r9d
  __int64 v15; // r8
  __int64 v16; // r12
  __int64 v17; // r11
  __int64 *v18; // r10
  __int64 v19; // rax
  __int64 *v21; // rdi
  int v22; // eax
  const char *v23; // rax
  unsigned __int64 v24; // rsi
  int v25; // eax
  unsigned int v26; // eax
  int v27; // eax
  __int32 v28; // eax
  int v29; // eax
  __int32 v30; // edx
  char v31; // al
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  char v38; // [rsp+38h] [rbp-2B8h]
  __int32 v39; // [rsp+38h] [rbp-2B8h]
  unsigned int v42; // [rsp+4Ch] [rbp-2A4h]
  __int16 v43; // [rsp+52h] [rbp-29Eh] BYREF
  __int16 v44; // [rsp+54h] [rbp-29Ch] BYREF
  __int16 v45; // [rsp+56h] [rbp-29Ah] BYREF
  int v46; // [rsp+58h] [rbp-298h] BYREF
  char v47; // [rsp+5Ch] [rbp-294h]
  __int64 *v48; // [rsp+60h] [rbp-290h] BYREF
  __int16 v49; // [rsp+68h] [rbp-288h]
  __int64 v50; // [rsp+70h] [rbp-280h] BYREF
  __int16 v51; // [rsp+78h] [rbp-278h]
  __int64 v52; // [rsp+80h] [rbp-270h] BYREF
  __int16 v53; // [rsp+88h] [rbp-268h]
  __int64 v54; // [rsp+90h] [rbp-260h] BYREF
  __int16 v55; // [rsp+98h] [rbp-258h]
  __int64 v56; // [rsp+A0h] [rbp-250h] BYREF
  __int16 v57; // [rsp+A8h] [rbp-248h]
  __int64 v58; // [rsp+B0h] [rbp-240h] BYREF
  __int16 v59; // [rsp+B8h] [rbp-238h]
  __int64 v60; // [rsp+C0h] [rbp-230h] BYREF
  __int16 v61; // [rsp+C8h] [rbp-228h]
  __int64 v62; // [rsp+D0h] [rbp-220h] BYREF
  __int16 v63; // [rsp+D8h] [rbp-218h]
  unsigned __int64 v64; // [rsp+E0h] [rbp-210h] BYREF
  __int16 v65; // [rsp+E8h] [rbp-208h]
  unsigned __int64 v66; // [rsp+F0h] [rbp-200h] BYREF
  __int16 v67; // [rsp+F8h] [rbp-1F8h]
  __int64 v68; // [rsp+100h] [rbp-1F0h] BYREF
  __int16 v69; // [rsp+108h] [rbp-1E8h]
  __int64 v70; // [rsp+110h] [rbp-1E0h] BYREF
  __int16 v71; // [rsp+118h] [rbp-1D8h]
  __int64 v72; // [rsp+120h] [rbp-1D0h] BYREF
  __int16 v73; // [rsp+128h] [rbp-1C8h]
  _QWORD v74[4]; // [rsp+130h] [rbp-1C0h] BYREF
  _QWORD v75[4]; // [rsp+150h] [rbp-1A0h] BYREF
  __int64 v76; // [rsp+170h] [rbp-180h] BYREF
  __int64 v77; // [rsp+178h] [rbp-178h]
  __int64 v78; // [rsp+180h] [rbp-170h]
  _QWORD v79[4]; // [rsp+190h] [rbp-160h] BYREF
  __int64 v80; // [rsp+1B0h] [rbp-140h] BYREF
  __int64 v81; // [rsp+1B8h] [rbp-138h]
  unsigned __int64 v82; // [rsp+1C0h] [rbp-130h]
  __int64 v83; // [rsp+1C8h] [rbp-128h]
  __m128i v84[2]; // [rsp+1D0h] [rbp-120h] BYREF
  char v85; // [rsp+1F0h] [rbp-100h]
  char v86; // [rsp+1F1h] [rbp-FFh]
  __m128i v87[2]; // [rsp+200h] [rbp-F0h] BYREF
  __int16 v88; // [rsp+220h] [rbp-D0h]
  __m128i v89; // [rsp+230h] [rbp-C0h] BYREF
  char *v90; // [rsp+240h] [rbp-B0h]
  __int16 v91; // [rsp+250h] [rbp-A0h]
  __m128i v92; // [rsp+260h] [rbp-90h] BYREF
  const char *v93; // [rsp+270h] [rbp-80h]
  __int16 v94; // [rsp+280h] [rbp-70h]
  __m128i v95; // [rsp+290h] [rbp-60h] BYREF
  char *v96; // [rsp+2A0h] [rbp-50h]
  __int64 v97; // [rsp+2A8h] [rbp-48h]
  __int16 v98; // [rsp+2B0h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 232);
  v49 = 256;
  v51 = 256;
  v55 = 256;
  v53 = 256;
  v74[2] = 0xFFFFFFFFLL;
  v57 = 256;
  v43 = 0;
  v44 = 1;
  v75[2] = 0xFFFFFFFFLL;
  v59 = 256;
  v48 = 0;
  v50 = 0;
  v52 = 0;
  v54 = 0;
  v74[0] = 0;
  v74[1] = 0;
  v56 = 0;
  v75[0] = 0;
  v75[1] = 0;
  v58 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 2;
  v79[2] = 0xFFFFFFFFLL;
  v65 = 256;
  v67 = 256;
  v45 = 0;
  v5 = a1 + 176;
  v69 = 256;
  v71 = 256;
  v79[0] = 0;
  v79[1] = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0xFFFFFFFF80000000LL;
  v83 = 0x7FFFFFFF;
  v46 = 0;
  v47 = 0;
  v60 = 0;
  v61 = 256;
  v62 = 0;
  v63 = 256;
  v64 = 0;
  v66 = 0;
  v68 = 0;
  v70 = 0;
  v72 = 0;
  v73 = 256;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v6 = sub_120AFE0(a1, 12, "expected '(' here");
  if ( (_BYTE)v6 )
    return 1;
  v7 = v6;
  v8 = *(_DWORD *)(a1 + 240);
  if ( v8 == 13 )
  {
    if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' here") )
      return 1;
    goto LABEL_17;
  }
  v38 = 0;
  v42 = 0;
  if ( v8 != 507 )
  {
LABEL_27:
    HIBYTE(v98) = 1;
    v23 = "expected field label here";
    goto LABEL_28;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "scope") )
    {
      v9 = sub_1225DC0(a1, (__int64)"scope", 5, (__int64)&v48);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "name") )
    {
      v9 = sub_120BB20(a1, "name", 4, (__int64)&v50);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "linkageName") )
    {
      v9 = sub_120BB20(a1, "linkageName", 11, (__int64)&v52);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "file") )
    {
      v9 = sub_1225DC0(a1, (__int64)"file", 4, (__int64)&v54);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "line") )
    {
      v9 = sub_1208380(a1, (__int64)"line", 4, (__int64)v74);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "type") )
    {
      v9 = sub_1225DC0(a1, (__int64)"type", 4, (__int64)&v56);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "isLocal") )
    {
      v9 = sub_1207D20(a1, (__int64)"isLocal", 7, (__int64)&v43);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "isDefinition") )
    {
      v9 = sub_1207D20(a1, (__int64)"isDefinition", 12, (__int64)&v44);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "scopeLine") )
    {
      v9 = sub_1208380(a1, (__int64)"scopeLine", 9, (__int64)v75);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "containingType") )
    {
      v9 = sub_1225DC0(a1, (__int64)"containingType", 14, (__int64)&v58);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "virtuality") )
    {
      if ( (_BYTE)v77 )
      {
        v98 = 1283;
        v95.m128i_i64[0] = (__int64)"field '";
        v96 = "virtuality";
        v97 = 10;
LABEL_47:
        v24 = *(_QWORD *)(a1 + 232);
        v92.m128i_i64[0] = (__int64)&v95;
        v93 = "' cannot be specified more than once";
        v94 = 770;
        sub_11FD800(v5, v24, (__int64)&v92, 1);
        return 1;
      }
      v25 = sub_1205200(v5);
      *(_DWORD *)(a1 + 240) = v25;
      if ( v25 != 529 )
      {
        if ( v25 != 515 )
        {
          HIBYTE(v98) = 1;
          v23 = "expected DWARF virtuality code";
          goto LABEL_28;
        }
        v26 = sub_E0A560(*(_QWORD *)(a1 + 248), *(_QWORD *)(a1 + 256));
        if ( v26 == -1 )
        {
          v93 = (const char *)(a1 + 248);
          v89.m128i_i64[0] = (__int64)"invalid DWARF virtuality code";
          v90 = " '";
          v91 = 771;
          v92.m128i_i64[0] = (__int64)&v89;
          v94 = 1026;
LABEL_66:
          v98 = 770;
          v95.m128i_i64[0] = (__int64)&v92;
          v96 = "'";
          goto LABEL_29;
        }
        LOBYTE(v77) = 1;
        v76 = v26;
        *(_DWORD *)(a1 + 240) = sub_1205200(v5);
        goto LABEL_7;
      }
      v9 = sub_1208110(a1, (__int64)"virtuality", 10, (__int64)&v76);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "virtualIndex") )
    {
      v9 = sub_1208450(a1, (__int64)"virtualIndex", 12, (__int64)v79);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "thisAdjustment") )
      break;
    if ( (_BYTE)v81 )
    {
      v98 = 1283;
      v95.m128i_i64[0] = (__int64)"field '";
      v96 = "thisAdjustment";
      v97 = 14;
      goto LABEL_47;
    }
    *(_DWORD *)(a1 + 240) = sub_1205200(v5);
    v9 = sub_1208D50(a1, (__int64)"thisAdjustment", 14, (__int64)&v80);
LABEL_6:
    if ( v9 )
      return 1;
LABEL_7:
    if ( *(_DWORD *)(a1 + 240) != 4 )
    {
      if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' here") )
        return 1;
      if ( v38 )
      {
        if ( (v42 & 8) != 0 )
        {
LABEL_11:
          if ( !a3 )
          {
            v95.m128i_i64[0] = (__int64)"missing 'distinct', required for !DISubprogram that is a Definition";
            v98 = 259;
            sub_11FD800(v5, v4, (__int64)&v95, 1);
            return 1;
          }
          v10 = v46;
          v11 = v80;
          v12 = v79[0];
          v13 = v75[0];
          v14 = v74[0];
          v15 = v54;
          v16 = v52;
          v17 = v50;
          v18 = v48;
          goto LABEL_13;
        }
LABEL_18:
        v10 = v46;
        v11 = v80;
        v12 = v79[0];
        v14 = v74[0];
        v15 = v54;
        v16 = v52;
        v17 = v50;
        v18 = v48;
        v21 = *(__int64 **)a1;
        v13 = v75[0];
        if ( !a3 )
        {
          v19 = sub_B07EA0(
                  v21,
                  v48,
                  v50,
                  v52,
                  v54,
                  v74[0],
                  v56,
                  v75[0],
                  v58,
                  v79[0],
                  v80,
                  v46,
                  v42,
                  v60,
                  v62,
                  v64,
                  v66,
                  v68,
                  v70,
                  v72,
                  0,
                  1);
          goto LABEL_14;
        }
LABEL_13:
        v19 = sub_B07EA0(
                *(__int64 **)a1,
                v18,
                v17,
                v16,
                v15,
                v14,
                v56,
                v13,
                v58,
                v12,
                v11,
                v10,
                v42,
                v60,
                v62,
                v64,
                v66,
                v68,
                v70,
                v72,
                1u,
                1);
LABEL_14:
        *a2 = v19;
        return v7;
      }
LABEL_17:
      v42 = sub_AF3490(v43, v44, v45, v76, 0);
      if ( (v42 & 8) != 0 )
        goto LABEL_11;
      goto LABEL_18;
    }
    v22 = sub_1205200(v5);
    *(_DWORD *)(a1 + 240) = v22;
    if ( v22 != 507 )
      goto LABEL_27;
  }
  if ( !(unsigned int)sub_2241AC0(a1 + 248, "flags") )
  {
    v9 = sub_120BE50(a1, (__int64)"flags", 5, (__int64)&v46);
    goto LABEL_6;
  }
  if ( (unsigned int)sub_2241AC0(a1 + 248, "spFlags") )
  {
    if ( (unsigned int)sub_2241AC0(a1 + 248, "isOptimized") )
    {
      if ( (unsigned int)sub_2241AC0(a1 + 248, "unit") )
      {
        if ( (unsigned int)sub_2241AC0(a1 + 248, "templateParams") )
        {
          if ( (unsigned int)sub_2241AC0(a1 + 248, "declaration") )
          {
            if ( (unsigned int)sub_2241AC0(a1 + 248, "retainedNodes") )
            {
              if ( (unsigned int)sub_2241AC0(a1 + 248, "thrownTypes") )
              {
                if ( (unsigned int)sub_2241AC0(a1 + 248, "annotations") )
                {
                  if ( (unsigned int)sub_2241AC0(a1 + 248, "targetFuncName") )
                  {
                    v87[0].m128i_i64[0] = a1 + 248;
                    v92.m128i_i64[0] = (__int64)"'";
                    v84[0].m128i_i64[0] = (__int64)"invalid field '";
                    v94 = 259;
                    v88 = 260;
                    v86 = 1;
                    v85 = 3;
                    sub_9C6370(&v89, v84, v87, v32, v33, v34);
                    sub_9C6370(&v95, &v89, &v92, v35, v36, v37);
                    sub_11FD800(v5, *(_QWORD *)(a1 + 232), (__int64)&v95, 1);
                    return 1;
                  }
                  v9 = sub_120BB20(a1, "targetFuncName", 14, (__int64)&v72);
                }
                else
                {
                  v9 = sub_1225DC0(a1, (__int64)"annotations", 11, (__int64)&v70);
                }
              }
              else
              {
                v9 = sub_1225DC0(a1, (__int64)"thrownTypes", 11, (__int64)&v68);
              }
            }
            else
            {
              v9 = sub_1225DC0(a1, (__int64)"retainedNodes", 13, (__int64)&v66);
            }
          }
          else
          {
            v9 = sub_1225DC0(a1, (__int64)"declaration", 11, (__int64)&v64);
          }
        }
        else
        {
          v9 = sub_1225DC0(a1, (__int64)"templateParams", 14, (__int64)&v62);
        }
      }
      else
      {
        v9 = sub_1225DC0(a1, (__int64)"unit", 4, (__int64)&v60);
      }
    }
    else
    {
      v9 = sub_1207D20(a1, (__int64)"isOptimized", 11, (__int64)&v45);
    }
    goto LABEL_6;
  }
  if ( v38 )
  {
    v98 = 1283;
    v95.m128i_i64[0] = (__int64)"field '";
    v96 = "spFlags";
    v97 = 7;
    goto LABEL_47;
  }
  v42 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(v5);
  while ( 2 )
  {
    v27 = *(_DWORD *)(a1 + 240);
    if ( v27 != 529 )
    {
      if ( v27 != 522 )
        break;
      v28 = sub_AF3560(*(_QWORD *)(a1 + 248), *(_QWORD *)(a1 + 256));
      if ( !v28 )
      {
        v93 = (const char *)(a1 + 248);
        v92.m128i_i64[0] = (__int64)"invalid subprogram debug info flag '";
        v94 = 1027;
        goto LABEL_66;
      }
      v39 = v28;
      v29 = sub_1205200(v5);
      v30 = v39;
      *(_DWORD *)(a1 + 240) = v29;
      goto LABEL_75;
    }
    if ( *(_BYTE *)(a1 + 332) )
    {
      v95.m128i_i32[0] = 0;
      v31 = sub_120BD00(a1, &v95);
      v30 = v95.m128i_i32[0];
      if ( v31 )
        return 1;
LABEL_75:
      v42 |= v30;
      if ( *(_DWORD *)(a1 + 240) != 15 )
      {
        v38 = 1;
        goto LABEL_7;
      }
      *(_DWORD *)(a1 + 240) = sub_1205200(v5);
      continue;
    }
    break;
  }
  HIBYTE(v98) = 1;
  v23 = "expected debug info flag";
LABEL_28:
  v95.m128i_i64[0] = (__int64)v23;
  LOBYTE(v98) = 3;
LABEL_29:
  sub_11FD800(v5, *(_QWORD *)(a1 + 232), (__int64)&v95, 1);
  return 1;
}

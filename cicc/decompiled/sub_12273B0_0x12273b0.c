// Function: sub_12273B0
// Address: 0x12273b0
//
__int64 __fastcall sub_12273B0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r14
  int v3; // eax
  char v4; // al
  unsigned __int64 v5; // r15
  unsigned int v6; // r12d
  const char *v7; // rax
  int v9; // eax
  const char *v10; // rax
  unsigned __int64 v11; // rsi
  int v12; // eax
  int v13; // edx
  const char *v14; // rax
  int v15; // eax
  int v16; // edx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int16 v23; // [rsp+8h] [rbp-278h] BYREF
  __int16 v24; // [rsp+Ah] [rbp-276h] BYREF
  __int16 v25; // [rsp+Ch] [rbp-274h] BYREF
  __int16 v26; // [rsp+Eh] [rbp-272h] BYREF
  __int64 v27; // [rsp+10h] [rbp-270h] BYREF
  __int16 v28; // [rsp+18h] [rbp-268h]
  __int64 v29; // [rsp+20h] [rbp-260h] BYREF
  __int16 v30; // [rsp+28h] [rbp-258h]
  __int64 v31; // [rsp+30h] [rbp-250h] BYREF
  __int16 v32; // [rsp+38h] [rbp-248h]
  __int64 v33; // [rsp+40h] [rbp-240h] BYREF
  __int16 v34; // [rsp+48h] [rbp-238h]
  __int64 v35; // [rsp+50h] [rbp-230h] BYREF
  __int16 v36; // [rsp+58h] [rbp-228h]
  __int64 v37; // [rsp+60h] [rbp-220h] BYREF
  __int16 v38; // [rsp+68h] [rbp-218h]
  __int64 v39; // [rsp+70h] [rbp-210h] BYREF
  __int16 v40; // [rsp+78h] [rbp-208h]
  __int64 v41; // [rsp+80h] [rbp-200h] BYREF
  __int16 v42; // [rsp+88h] [rbp-1F8h]
  __int64 v43; // [rsp+90h] [rbp-1F0h] BYREF
  __int16 v44; // [rsp+98h] [rbp-1E8h]
  __int64 v45; // [rsp+A0h] [rbp-1E0h] BYREF
  __int16 v46; // [rsp+A8h] [rbp-1D8h]
  __int64 v47; // [rsp+B0h] [rbp-1D0h] BYREF
  __int16 v48; // [rsp+B8h] [rbp-1C8h]
  __int64 v49; // [rsp+C0h] [rbp-1C0h] BYREF
  __int64 v50; // [rsp+C8h] [rbp-1B8h]
  __int64 v51; // [rsp+D0h] [rbp-1B0h]
  _QWORD v52[4]; // [rsp+E0h] [rbp-1A0h] BYREF
  __int64 v53; // [rsp+100h] [rbp-180h] BYREF
  __int64 v54; // [rsp+108h] [rbp-178h]
  __int64 v55; // [rsp+110h] [rbp-170h]
  __int64 v56[4]; // [rsp+120h] [rbp-160h] BYREF
  __int64 v57; // [rsp+140h] [rbp-140h] BYREF
  __int64 v58; // [rsp+148h] [rbp-138h]
  __int64 v59; // [rsp+150h] [rbp-130h]
  __m128i v60[2]; // [rsp+160h] [rbp-120h] BYREF
  char v61; // [rsp+180h] [rbp-100h]
  char v62; // [rsp+181h] [rbp-FFh]
  __m128i v63; // [rsp+190h] [rbp-F0h] BYREF
  __int16 v64; // [rsp+1B0h] [rbp-D0h]
  __m128i v65; // [rsp+1C0h] [rbp-C0h] BYREF
  char *v66; // [rsp+1D0h] [rbp-B0h]
  __int16 v67; // [rsp+1E0h] [rbp-A0h]
  __m128i v68; // [rsp+1F0h] [rbp-90h] BYREF
  const char *v69; // [rsp+200h] [rbp-80h]
  __int16 v70; // [rsp+210h] [rbp-70h]
  __m128i v71; // [rsp+220h] [rbp-60h] BYREF
  char *v72; // [rsp+230h] [rbp-50h]
  __int64 v73; // [rsp+238h] [rbp-48h]
  __int16 v74; // [rsp+240h] [rbp-40h]

  v52[2] = 0xFFFFFFFFLL;
  v30 = 256;
  v28 = 0;
  v23 = 0;
  v32 = 256;
  v34 = 256;
  v36 = 256;
  v40 = 256;
  v2 = a1 + 176;
  v49 = 0;
  v50 = 0;
  v51 = 0xFFFF;
  v27 = 0;
  v29 = 0;
  v31 = 0;
  v52[0] = 0;
  v52[1] = 0;
  v33 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 3;
  v35 = 0;
  v37 = 0;
  v38 = 256;
  v39 = 0;
  v41 = 0;
  v42 = 256;
  v44 = 256;
  v24 = 1;
  v25 = 0;
  v26 = 0;
  v48 = 256;
  v43 = 0;
  v56[0] = 0;
  v56[1] = 0;
  v56[2] = -1;
  v57 = 0;
  v58 = 0;
  v59 = 3;
  v45 = 0;
  v46 = 256;
  v47 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  else
  {
    v3 = *(_DWORD *)(a1 + 240);
    if ( v3 != 13 )
    {
      if ( v3 != 507 )
      {
LABEL_21:
        HIBYTE(v74) = 1;
        v10 = "expected field label here";
LABEL_22:
        v71.m128i_i64[0] = (__int64)v10;
        LOBYTE(v74) = 3;
        goto LABEL_23;
      }
      while ( 1 )
      {
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "language") )
        {
          v4 = sub_1208700(a1, (__int64)"language", 8, (__int64)&v49);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "file") )
        {
          v4 = sub_1225DC0(a1, (__int64)"file", 4, (__int64)&v27);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "producer") )
        {
          v4 = sub_120BB20(a1, "producer", 8, (__int64)&v29);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "isOptimized") )
        {
          v4 = sub_1207D20(a1, (__int64)"isOptimized", 11, (__int64)&v23);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "flags") )
        {
          v4 = sub_120BB20(a1, "flags", 5, (__int64)&v31);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "runtimeVersion") )
        {
          v4 = sub_1208450(a1, (__int64)"runtimeVersion", 14, (__int64)v52);
          goto LABEL_6;
        }
        if ( !(unsigned int)sub_2241AC0(a1 + 248, "splitDebugFilename") )
        {
          v4 = sub_120BB20(a1, "splitDebugFilename", 18, (__int64)&v33);
          goto LABEL_6;
        }
        if ( (unsigned int)sub_2241AC0(a1 + 248, "emissionKind") )
        {
          if ( !(unsigned int)sub_2241AC0(a1 + 248, "enums") )
          {
            v4 = sub_1225DC0(a1, (__int64)"enums", 5, (__int64)&v35);
            goto LABEL_6;
          }
          if ( !(unsigned int)sub_2241AC0(a1 + 248, "retainedTypes") )
          {
            v4 = sub_1225DC0(a1, (__int64)"retainedTypes", 13, (__int64)&v37);
            goto LABEL_6;
          }
          if ( !(unsigned int)sub_2241AC0(a1 + 248, "globals") )
          {
            v4 = sub_1225DC0(a1, (__int64)"globals", 7, (__int64)&v39);
            goto LABEL_6;
          }
          if ( !(unsigned int)sub_2241AC0(a1 + 248, "imports") )
          {
            v4 = sub_1225DC0(a1, (__int64)"imports", 7, (__int64)&v41);
            goto LABEL_6;
          }
          if ( !(unsigned int)sub_2241AC0(a1 + 248, "macros") )
          {
            v4 = sub_1225DC0(a1, (__int64)"macros", 6, (__int64)&v43);
            goto LABEL_6;
          }
          if ( !(unsigned int)sub_2241AC0(a1 + 248, "dwoId") )
          {
            v4 = sub_1208450(a1, (__int64)"dwoId", 5, (__int64)v56);
            goto LABEL_6;
          }
          if ( !(unsigned int)sub_2241AC0(a1 + 248, "splitDebugInlining") )
          {
            v4 = sub_1207D20(a1, (__int64)"splitDebugInlining", 18, (__int64)&v24);
            goto LABEL_6;
          }
          if ( !(unsigned int)sub_2241AC0(a1 + 248, "debugInfoForProfiling") )
          {
            v4 = sub_1207D20(a1, (__int64)"debugInfoForProfiling", 21, (__int64)&v25);
            goto LABEL_6;
          }
          if ( (unsigned int)sub_2241AC0(a1 + 248, "nameTableKind") )
          {
            if ( (unsigned int)sub_2241AC0(a1 + 248, "rangesBaseAddress") )
            {
              if ( (unsigned int)sub_2241AC0(a1 + 248, "sysroot") )
              {
                if ( (unsigned int)sub_2241AC0(a1 + 248, "sdk") )
                {
                  v63.m128i_i64[0] = a1 + 248;
                  v68.m128i_i64[0] = (__int64)"'";
                  v60[0].m128i_i64[0] = (__int64)"invalid field '";
                  v70 = 259;
                  v64 = 260;
                  v62 = 1;
                  v61 = 3;
                  sub_9C6370(&v65, v60, &v63, v17, v18, v19);
                  sub_9C6370(&v71, &v65, &v68, v20, v21, v22);
                  sub_11FD800(v2, *(_QWORD *)(a1 + 232), (__int64)&v71, 1);
                  return 1;
                }
                v4 = sub_120BB20(a1, "sdk", 3, (__int64)&v47);
              }
              else
              {
                v4 = sub_120BB20(a1, "sysroot", 7, (__int64)&v45);
              }
            }
            else
            {
              v4 = sub_1207D20(a1, (__int64)"rangesBaseAddress", 17, (__int64)&v26);
            }
            goto LABEL_6;
          }
          if ( (_BYTE)v58 )
          {
            v74 = 1283;
            v71.m128i_i64[0] = (__int64)"field '";
            v72 = "nameTableKind";
            v73 = 13;
            goto LABEL_36;
          }
          v15 = sub_1205200(v2);
          *(_DWORD *)(a1 + 240) = v15;
          if ( v15 == 529 )
          {
            v4 = sub_1208110(a1, (__int64)"nameTableKind", 13, (__int64)&v57);
LABEL_6:
            if ( v4 )
              return 1;
            goto LABEL_7;
          }
          if ( v15 != 519 )
          {
            HIBYTE(v74) = 1;
            v10 = "expected nameTable kind";
            goto LABEL_22;
          }
          v63.m128i_i64[0] = sub_AF3310(*(_QWORD *)(a1 + 248), *(_QWORD *)(a1 + 256), v16);
          if ( !v63.m128i_i8[4] )
          {
            v14 = "invalid nameTable kind";
LABEL_48:
            v65.m128i_i64[0] = (__int64)v14;
            v66 = " '";
            v68.m128i_i64[0] = (__int64)&v65;
            v71.m128i_i64[0] = (__int64)&v68;
            v67 = 771;
            v69 = (const char *)(a1 + 248);
            v70 = 1026;
            v72 = "'";
            v74 = 770;
LABEL_23:
            sub_11FD800(v2, *(_QWORD *)(a1 + 232), (__int64)&v71, 1);
            return 1;
          }
          LOBYTE(v58) = 1;
          v57 = v63.m128i_u32[0];
          *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        }
        else
        {
          if ( (_BYTE)v54 )
          {
            v73 = 12;
            v71.m128i_i64[0] = (__int64)"field '";
            v74 = 1283;
            v72 = "emissionKind";
LABEL_36:
            v11 = *(_QWORD *)(a1 + 232);
            v68.m128i_i64[0] = (__int64)&v71;
            v69 = "' cannot be specified more than once";
            v70 = 770;
            sub_11FD800(v2, v11, (__int64)&v68, 1);
            return 1;
          }
          v12 = sub_1205200(v2);
          *(_DWORD *)(a1 + 240) = v12;
          if ( v12 == 529 )
          {
            v4 = sub_1208110(a1, (__int64)"emissionKind", 12, (__int64)&v53);
            goto LABEL_6;
          }
          if ( v12 != 518 )
          {
            HIBYTE(v74) = 1;
            v10 = "expected emission kind";
            goto LABEL_22;
          }
          v63.m128i_i64[0] = sub_AF3210(*(_QWORD *)(a1 + 248), *(_QWORD *)(a1 + 256), v13);
          if ( !v63.m128i_i8[4] )
          {
            v14 = "invalid emission kind";
            goto LABEL_48;
          }
          LOBYTE(v54) = 1;
          v53 = v63.m128i_u32[0];
          *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        }
LABEL_7:
        if ( *(_DWORD *)(a1 + 240) != 4 )
          break;
        v9 = sub_1205200(v2);
        *(_DWORD *)(a1 + 240) = v9;
        if ( v9 != 507 )
          goto LABEL_21;
      }
    }
    v5 = *(_QWORD *)(a1 + 232);
    v6 = sub_120AFE0(a1, 13, "expected ')' here");
    if ( (_BYTE)v6 )
      return 1;
    if ( !(_BYTE)v50 )
    {
      HIBYTE(v74) = 1;
      v7 = "missing required field 'language'";
      goto LABEL_12;
    }
    if ( !(_BYTE)v28 )
    {
      HIBYTE(v74) = 1;
      v7 = "missing required field 'file'";
LABEL_12:
      v71.m128i_i64[0] = (__int64)v7;
      LOBYTE(v74) = 3;
      sub_11FD800(v2, v5, (__int64)&v71, 1);
      return 1;
    }
    *a2 = sub_AF30C0(
            *(_QWORD *)a1,
            v49,
            v27,
            v29,
            v23,
            v31,
            v52[0],
            v33,
            v53,
            v35,
            v37,
            v39,
            v41,
            v43,
            v56[0],
            v24,
            v25,
            v57,
            v26,
            v45,
            v47,
            1u);
  }
  return v6;
}

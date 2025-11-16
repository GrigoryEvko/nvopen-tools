// Function: sub_38A4F40
// Address: 0x38a4f40
//
__int64 __fastcall sub_38A4F40(__int64 a1, __int64 *a2, double a3, double a4, double a5)
{
  __int64 v6; // r12
  int v7; // eax
  char v8; // al
  unsigned __int64 v9; // r14
  unsigned int v10; // r13d
  const char *v11; // rax
  int v13; // eax
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rsi
  int v16; // eax
  __m128i v17; // kr00_16
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rsi
  __int16 v20; // [rsp+18h] [rbp-1E8h] BYREF
  __int16 v21; // [rsp+1Ah] [rbp-1E6h] BYREF
  __int16 v22; // [rsp+1Ch] [rbp-1E4h] BYREF
  __int16 v23; // [rsp+1Eh] [rbp-1E2h] BYREF
  __int64 v24; // [rsp+20h] [rbp-1E0h] BYREF
  __int16 v25; // [rsp+28h] [rbp-1D8h]
  __int64 v26; // [rsp+30h] [rbp-1D0h] BYREF
  __int16 v27; // [rsp+38h] [rbp-1C8h]
  __int64 v28; // [rsp+40h] [rbp-1C0h] BYREF
  __int16 v29; // [rsp+48h] [rbp-1B8h]
  __int64 v30; // [rsp+50h] [rbp-1B0h] BYREF
  __int16 v31; // [rsp+58h] [rbp-1A8h]
  __int64 v32; // [rsp+60h] [rbp-1A0h] BYREF
  __int16 v33; // [rsp+68h] [rbp-198h]
  __int64 v34; // [rsp+70h] [rbp-190h] BYREF
  __int16 v35; // [rsp+78h] [rbp-188h]
  __int64 v36; // [rsp+80h] [rbp-180h] BYREF
  __int16 v37; // [rsp+88h] [rbp-178h]
  __int64 v38; // [rsp+90h] [rbp-170h] BYREF
  __int16 v39; // [rsp+98h] [rbp-168h]
  __int64 v40; // [rsp+A0h] [rbp-160h] BYREF
  __int16 v41; // [rsp+A8h] [rbp-158h]
  __int64 v42; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v43; // [rsp+B8h] [rbp-148h]
  __int64 v44; // [rsp+C0h] [rbp-140h]
  _QWORD v45[4]; // [rsp+D0h] [rbp-130h] BYREF
  __int64 v46; // [rsp+F0h] [rbp-110h] BYREF
  __int64 v47; // [rsp+F8h] [rbp-108h]
  __int64 v48; // [rsp+100h] [rbp-100h]
  __int64 v49[4]; // [rsp+110h] [rbp-F0h] BYREF
  __m128i v50; // [rsp+130h] [rbp-D0h] BYREF
  char v51; // [rsp+140h] [rbp-C0h]
  char v52; // [rsp+141h] [rbp-BFh]
  __m128i v53; // [rsp+150h] [rbp-B0h] BYREF
  __int16 v54; // [rsp+160h] [rbp-A0h]
  __m128i v55; // [rsp+170h] [rbp-90h] BYREF
  __int16 v56; // [rsp+180h] [rbp-80h]
  __m128i v57; // [rsp+190h] [rbp-70h] BYREF
  __int16 v58; // [rsp+1A0h] [rbp-60h]
  __m128i v59; // [rsp+1B0h] [rbp-50h] BYREF
  __int16 v60; // [rsp+1C0h] [rbp-40h]

  v45[2] = 0xFFFFFFFFLL;
  v27 = 256;
  v25 = 0;
  v20 = 0;
  v29 = 256;
  v31 = 256;
  v33 = 256;
  v35 = 256;
  v6 = a1 + 8;
  v42 = 0;
  v43 = 0;
  v44 = 0xFFFF;
  v24 = 0;
  v26 = 0;
  v28 = 0;
  v45[0] = 0;
  v45[1] = 0;
  v30 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 3;
  v32 = 0;
  v34 = 0;
  v36 = 0;
  v37 = 256;
  v38 = 0;
  v39 = 256;
  v41 = 256;
  v21 = 1;
  v22 = 0;
  v40 = 0;
  v49[0] = 0;
  v49[1] = 0;
  v49[2] = -1;
  v23 = 0;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v7 = *(_DWORD *)(a1 + 64);
  if ( v7 == 13 )
    goto LABEL_8;
  if ( v7 == 372 )
  {
    while ( 1 )
    {
      if ( !sub_2241AC0(a1 + 72, "language") )
      {
        v8 = sub_3889810(a1, (__int64)"language", 8, (__int64)&v42);
        goto LABEL_6;
      }
      if ( !sub_2241AC0(a1 + 72, "file") )
      {
        v8 = sub_38A29E0(a1, (__int64)"file", 4, (__int64)&v24, a3, a4, a5);
        goto LABEL_6;
      }
      if ( !sub_2241AC0(a1 + 72, "producer") )
      {
        v8 = sub_388B8F0(a1, (__int64)"producer", 8, (__int64)&v26);
        goto LABEL_6;
      }
      if ( !sub_2241AC0(a1 + 72, "isOptimized") )
      {
        v8 = sub_3887760(a1, (__int64)"isOptimized", 11, (__int64)&v20);
        goto LABEL_6;
      }
      if ( !sub_2241AC0(a1 + 72, "flags") )
      {
        v8 = sub_388B8F0(a1, (__int64)"flags", 5, (__int64)&v28);
        goto LABEL_6;
      }
      if ( !sub_2241AC0(a1 + 72, "runtimeVersion") )
      {
        v8 = sub_3889510(a1, (__int64)"runtimeVersion", 14, (__int64)v45);
        goto LABEL_6;
      }
      if ( !sub_2241AC0(a1 + 72, "splitDebugFilename") )
      {
        v8 = sub_388B8F0(a1, (__int64)"splitDebugFilename", 18, (__int64)&v30);
        goto LABEL_6;
      }
      if ( sub_2241AC0(a1 + 72, "emissionKind") )
        break;
      v53.m128i_i64[1] = 12;
      v53.m128i_i64[0] = (__int64)"emissionKind";
      if ( (_BYTE)v47 )
      {
        v15 = *(_QWORD *)(a1 + 56);
        v60 = 1283;
        v59.m128i_i64[0] = (__int64)"field '";
        v59.m128i_i64[1] = (__int64)&v53;
        v57.m128i_i64[0] = (__int64)&v59;
        v58 = 770;
        v57.m128i_i64[1] = (__int64)"' cannot be specified more than once";
        v8 = sub_38814C0(v6, v15, (__int64)&v57);
        goto LABEL_6;
      }
      v16 = sub_3887100(v6);
      v17 = v53;
      *(_DWORD *)(a1 + 64) = v16;
      if ( v16 == 390 )
      {
        v8 = sub_3889300(a1, v17.m128i_i64[0], v17.m128i_i64[1], (__int64)&v46);
        goto LABEL_6;
      }
      if ( v16 != 383 )
      {
        v18 = *(_QWORD *)(a1 + 56);
        v60 = 259;
        v59.m128i_i64[0] = (__int64)"expected emission kind";
        v8 = sub_38814C0(v6, v18, (__int64)&v59);
        goto LABEL_6;
      }
      sub_15B0EE0((__int64)&v50, *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80));
      if ( !v50.m128i_i8[4] )
      {
        v19 = *(_QWORD *)(a1 + 56);
        v60 = 771;
        v59.m128i_i64[0] = (__int64)"invalid emission kind";
        v59.m128i_i64[1] = (__int64)" '";
        v57.m128i_i64[0] = (__int64)&v59;
        v57.m128i_i64[1] = a1 + 72;
        v58 = 1026;
        v55.m128i_i64[0] = (__int64)&v57;
        v55.m128i_i64[1] = (__int64)"'";
        v56 = 770;
        v8 = sub_38814C0(v6, v19, (__int64)&v55);
        goto LABEL_6;
      }
      LOBYTE(v47) = 1;
      v46 = v50.m128i_u32[0];
      *(_DWORD *)(a1 + 64) = sub_3887100(v6);
LABEL_7:
      if ( *(_DWORD *)(a1 + 64) != 4 )
        goto LABEL_8;
      v13 = sub_3887100(v6);
      *(_DWORD *)(a1 + 64) = v13;
      if ( v13 != 372 )
        goto LABEL_16;
    }
    if ( sub_2241AC0(a1 + 72, "enums") )
    {
      if ( sub_2241AC0(a1 + 72, "retainedTypes") )
      {
        if ( sub_2241AC0(a1 + 72, "globals") )
        {
          if ( sub_2241AC0(a1 + 72, "imports") )
          {
            if ( sub_2241AC0(a1 + 72, "macros") )
            {
              if ( sub_2241AC0(a1 + 72, "dwoId") )
              {
                if ( sub_2241AC0(a1 + 72, "splitDebugInlining") )
                {
                  if ( sub_2241AC0(a1 + 72, "debugInfoForProfiling") )
                  {
                    if ( sub_2241AC0(a1 + 72, "gnuPubnames") )
                    {
                      v50.m128i_i64[0] = (__int64)"invalid field '";
                      v57.m128i_i64[0] = (__int64)"'";
                      v58 = 259;
                      v54 = 260;
                      v53.m128i_i64[0] = a1 + 72;
                      v52 = 1;
                      v51 = 3;
                      sub_14EC200(&v55, &v50, &v53);
                      sub_14EC200(&v59, &v55, &v57);
                      v8 = sub_38814C0(v6, *(_QWORD *)(a1 + 56), (__int64)&v59);
                    }
                    else
                    {
                      v8 = sub_3887760(a1, (__int64)"gnuPubnames", 11, (__int64)&v23);
                    }
                  }
                  else
                  {
                    v8 = sub_3887760(a1, (__int64)"debugInfoForProfiling", 21, (__int64)&v22);
                  }
                }
                else
                {
                  v8 = sub_3887760(a1, (__int64)"splitDebugInlining", 18, (__int64)&v21);
                }
              }
              else
              {
                v8 = sub_3889510(a1, (__int64)"dwoId", 5, (__int64)v49);
              }
            }
            else
            {
              v8 = sub_38A29E0(a1, (__int64)"macros", 6, (__int64)&v40, a3, a4, a5);
            }
          }
          else
          {
            v8 = sub_38A29E0(a1, (__int64)"imports", 7, (__int64)&v38, a3, a4, a5);
          }
        }
        else
        {
          v8 = sub_38A29E0(a1, (__int64)"globals", 7, (__int64)&v36, a3, a4, a5);
        }
      }
      else
      {
        v8 = sub_38A29E0(a1, (__int64)"retainedTypes", 13, (__int64)&v34, a3, a4, a5);
      }
    }
    else
    {
      v8 = sub_38A29E0(a1, (__int64)"enums", 5, (__int64)&v32, a3, a4, a5);
    }
LABEL_6:
    if ( v8 )
      return 1;
    goto LABEL_7;
  }
LABEL_16:
  v14 = *(_QWORD *)(a1 + 56);
  v60 = 259;
  v59.m128i_i64[0] = (__int64)"expected field label here";
  if ( (unsigned __int8)sub_38814C0(v6, v14, (__int64)&v59) )
  {
    return 1;
  }
  else
  {
LABEL_8:
    v9 = *(_QWORD *)(a1 + 56);
    v10 = sub_388AF10(a1, 13, "expected ')' here");
    if ( !(_BYTE)v10 )
    {
      if ( (_BYTE)v43 )
      {
        if ( (_BYTE)v25 )
        {
          *a2 = sub_15B0DC0(
                  *(_QWORD *)a1,
                  v42,
                  v24,
                  v26,
                  v20,
                  v28,
                  v45[0],
                  v30,
                  v46,
                  v32,
                  v34,
                  v36,
                  v38,
                  v40,
                  v49[0],
                  v21,
                  v22,
                  v23,
                  1);
          return v10;
        }
        HIBYTE(v60) = 1;
        v11 = "missing required field 'file'";
      }
      else
      {
        HIBYTE(v60) = 1;
        v11 = "missing required field 'language'";
      }
      v59.m128i_i64[0] = (__int64)v11;
      LOBYTE(v60) = 3;
      return (unsigned int)sub_38814C0(v6, v9, (__int64)&v59);
    }
  }
  return v10;
}

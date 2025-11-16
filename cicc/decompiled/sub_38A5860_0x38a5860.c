// Function: sub_38A5860
// Address: 0x38a5860
//
__int64 __fastcall sub_38A5860(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  unsigned __int64 v7; // r13
  __int64 v8; // r12
  int v9; // eax
  char v10; // al
  unsigned int v11; // r14d
  char v12; // r8
  unsigned int v13; // edi
  unsigned int v14; // esi
  unsigned int v15; // ecx
  char v16; // r10
  int v17; // r9d
  __int64 v18; // rbx
  __int64 v19; // r11
  __int64 v20; // r13
  __int64 v21; // r12
  __int64 v22; // rax
  int v24; // eax
  unsigned __int64 v25; // rsi
  unsigned __int64 v26; // rsi
  int v27; // eax
  __m128i v28; // kr00_16
  unsigned __int64 v29; // rsi
  unsigned int v30; // eax
  unsigned __int64 v31; // rsi
  _QWORD *v32; // [rsp+30h] [rbp-260h]
  unsigned int v33; // [rsp+40h] [rbp-250h]
  int v34; // [rsp+48h] [rbp-248h]
  __int16 v37; // [rsp+62h] [rbp-22Eh] BYREF
  __int16 v38; // [rsp+64h] [rbp-22Ch] BYREF
  __int16 v39; // [rsp+66h] [rbp-22Ah] BYREF
  unsigned int v40; // [rsp+68h] [rbp-228h] BYREF
  char v41; // [rsp+6Ch] [rbp-224h]
  __int64 v42; // [rsp+70h] [rbp-220h] BYREF
  __int16 v43; // [rsp+78h] [rbp-218h]
  __int64 v44; // [rsp+80h] [rbp-210h] BYREF
  __int16 v45; // [rsp+88h] [rbp-208h]
  __int64 v46; // [rsp+90h] [rbp-200h] BYREF
  __int16 v47; // [rsp+98h] [rbp-1F8h]
  __int64 v48; // [rsp+A0h] [rbp-1F0h] BYREF
  __int16 v49; // [rsp+A8h] [rbp-1E8h]
  __int64 v50; // [rsp+B0h] [rbp-1E0h] BYREF
  __int16 v51; // [rsp+B8h] [rbp-1D8h]
  __int64 v52; // [rsp+C0h] [rbp-1D0h] BYREF
  __int16 v53; // [rsp+C8h] [rbp-1C8h]
  __int64 v54; // [rsp+D0h] [rbp-1C0h] BYREF
  __int16 v55; // [rsp+D8h] [rbp-1B8h]
  __int64 v56; // [rsp+E0h] [rbp-1B0h] BYREF
  __int16 v57; // [rsp+E8h] [rbp-1A8h]
  unsigned __int64 v58; // [rsp+F0h] [rbp-1A0h] BYREF
  __int16 v59; // [rsp+F8h] [rbp-198h]
  unsigned __int64 v60; // [rsp+100h] [rbp-190h] BYREF
  __int16 v61; // [rsp+108h] [rbp-188h]
  __int64 v62; // [rsp+110h] [rbp-180h] BYREF
  __int16 v63; // [rsp+118h] [rbp-178h]
  _QWORD v64[4]; // [rsp+120h] [rbp-170h] BYREF
  _QWORD v65[4]; // [rsp+140h] [rbp-150h] BYREF
  __int64 v66; // [rsp+160h] [rbp-130h] BYREF
  __int64 v67; // [rsp+168h] [rbp-128h]
  __int64 v68; // [rsp+170h] [rbp-120h]
  _QWORD v69[4]; // [rsp+180h] [rbp-110h] BYREF
  __m128i v70; // [rsp+1A0h] [rbp-F0h] BYREF
  char v71; // [rsp+1B0h] [rbp-E0h]
  char v72; // [rsp+1B1h] [rbp-DFh]
  __m128i v73; // [rsp+1C0h] [rbp-D0h] BYREF
  __int16 v74; // [rsp+1D0h] [rbp-C0h]
  __m128i v75; // [rsp+1E0h] [rbp-B0h] BYREF
  __int16 v76; // [rsp+1F0h] [rbp-A0h]
  __m128i v77; // [rsp+200h] [rbp-90h] BYREF
  __int16 v78; // [rsp+210h] [rbp-80h]
  __m128i v79; // [rsp+220h] [rbp-70h] BYREF
  __int16 v80; // [rsp+230h] [rbp-60h]
  _QWORD v81[10]; // [rsp+240h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(a1 + 56);
  v43 = 256;
  v45 = 256;
  v49 = 256;
  v47 = 256;
  v64[2] = 0xFFFFFFFFLL;
  v51 = 256;
  v37 = 0;
  v38 = 1;
  v65[2] = 0xFFFFFFFFLL;
  v53 = 256;
  v42 = 0;
  v44 = 0;
  v46 = 0;
  v48 = 0;
  v64[0] = 0;
  v64[1] = 0;
  v50 = 0;
  v65[0] = 0;
  v65[1] = 0;
  v52 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 2;
  v69[2] = 0xFFFFFFFFLL;
  v55 = 256;
  v8 = a1 + 8;
  v59 = 256;
  v61 = 256;
  v69[0] = 0;
  v69[1] = 0;
  v81[0] = 0;
  v81[1] = 0;
  v81[2] = 0xFFFFFFFF80000000LL;
  v81[3] = 0x7FFFFFFF;
  v40 = 0;
  v41 = 0;
  v39 = 0;
  v54 = 0;
  v56 = 0;
  v57 = 256;
  v58 = 0;
  v60 = 0;
  v62 = 0;
  v63 = 256;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v9 = *(_DWORD *)(a1 + 64);
  if ( v9 != 13 )
  {
    if ( v9 == 372 )
    {
      while ( 1 )
      {
        if ( !sub_2241AC0(a1 + 72, "scope") )
        {
          v10 = sub_38A29E0(a1, (__int64)"scope", 5, (__int64)&v42, a4, a5, a6);
          goto LABEL_6;
        }
        if ( !sub_2241AC0(a1 + 72, "name") )
        {
          v10 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v44);
          goto LABEL_6;
        }
        if ( !sub_2241AC0(a1 + 72, "linkageName") )
        {
          v10 = sub_388B8F0(a1, (__int64)"linkageName", 11, (__int64)&v46);
          goto LABEL_6;
        }
        if ( !sub_2241AC0(a1 + 72, "file") )
        {
          v10 = sub_38A29E0(a1, (__int64)"file", 4, (__int64)&v48, a4, a5, a6);
          goto LABEL_6;
        }
        if ( !sub_2241AC0(a1 + 72, "line") )
        {
          v10 = sub_38895C0(a1, (__int64)"line", 4, (__int64)v64);
          goto LABEL_6;
        }
        if ( !sub_2241AC0(a1 + 72, "type") )
        {
          v10 = sub_38A29E0(a1, (__int64)"type", 4, (__int64)&v50, a4, a5, a6);
          goto LABEL_6;
        }
        if ( !sub_2241AC0(a1 + 72, "isLocal") )
        {
          v10 = sub_3887760(a1, (__int64)"isLocal", 7, (__int64)&v37);
          goto LABEL_6;
        }
        if ( !sub_2241AC0(a1 + 72, "isDefinition") )
        {
          v10 = sub_3887760(a1, (__int64)"isDefinition", 12, (__int64)&v38);
          goto LABEL_6;
        }
        if ( !sub_2241AC0(a1 + 72, "scopeLine") )
        {
          v10 = sub_38895C0(a1, (__int64)"scopeLine", 9, (__int64)v65);
          goto LABEL_6;
        }
        if ( !sub_2241AC0(a1 + 72, "containingType") )
        {
          v10 = sub_38A29E0(a1, (__int64)"containingType", 14, (__int64)&v52, a4, a5, a6);
          goto LABEL_6;
        }
        if ( sub_2241AC0(a1 + 72, "virtuality") )
          break;
        v73.m128i_i64[1] = 10;
        v73.m128i_i64[0] = (__int64)"virtuality";
        if ( (_BYTE)v67 )
        {
          v26 = *(_QWORD *)(a1 + 56);
          v80 = 1283;
          v79.m128i_i64[0] = (__int64)"field '";
          v79.m128i_i64[1] = (__int64)&v73;
          v77.m128i_i64[0] = (__int64)&v79;
          v77.m128i_i64[1] = (__int64)"' cannot be specified more than once";
          v78 = 770;
          v10 = sub_38814C0(v8, v26, (__int64)&v77);
          goto LABEL_6;
        }
        v27 = sub_3887100(v8);
        v28 = v73;
        *(_DWORD *)(a1 + 64) = v27;
        if ( v27 == 390 )
        {
          v10 = sub_3889300(a1, v28.m128i_i64[0], v28.m128i_i64[1], (__int64)&v66);
          goto LABEL_6;
        }
        if ( v27 != 380 )
        {
          v29 = *(_QWORD *)(a1 + 56);
          v80 = 259;
          v79.m128i_i64[0] = (__int64)"expected DWARF virtuality code";
          v10 = sub_38814C0(v8, v29, (__int64)&v79);
          goto LABEL_6;
        }
        v30 = sub_14E7710(*(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80));
        if ( v30 == -1 )
        {
          v31 = *(_QWORD *)(a1 + 56);
          v80 = 771;
          v79.m128i_i64[0] = (__int64)"invalid DWARF virtuality code";
          v79.m128i_i64[1] = (__int64)" '";
          v77.m128i_i64[0] = (__int64)&v79;
          v77.m128i_i64[1] = a1 + 72;
          v78 = 1026;
          v75.m128i_i64[0] = (__int64)&v77;
          v75.m128i_i64[1] = (__int64)"'";
          v76 = 770;
          v10 = sub_38814C0(v8, v31, (__int64)&v75);
          goto LABEL_6;
        }
        LOBYTE(v67) = 1;
        v66 = v30;
        *(_DWORD *)(a1 + 64) = sub_3887100(v8);
LABEL_7:
        if ( *(_DWORD *)(a1 + 64) != 4 )
          goto LABEL_8;
        v24 = sub_3887100(v8);
        *(_DWORD *)(a1 + 64) = v24;
        if ( v24 != 372 )
          goto LABEL_18;
      }
      if ( sub_2241AC0(a1 + 72, "virtualIndex") )
      {
        if ( sub_2241AC0(a1 + 72, "thisAdjustment") )
        {
          if ( sub_2241AC0(a1 + 72, "flags") )
          {
            if ( sub_2241AC0(a1 + 72, "isOptimized") )
            {
              if ( sub_2241AC0(a1 + 72, "unit") )
              {
                if ( sub_2241AC0(a1 + 72, "templateParams") )
                {
                  if ( sub_2241AC0(a1 + 72, "declaration") )
                  {
                    if ( sub_2241AC0(a1 + 72, "retainedNodes") )
                    {
                      if ( sub_2241AC0(a1 + 72, "thrownTypes") )
                      {
                        v70.m128i_i64[0] = (__int64)"invalid field '";
                        v77.m128i_i64[0] = (__int64)"'";
                        v78 = 259;
                        v74 = 260;
                        v73.m128i_i64[0] = a1 + 72;
                        v72 = 1;
                        v71 = 3;
                        sub_14EC200(&v75, &v70, &v73);
                        sub_14EC200(&v79, &v75, &v77);
                        v10 = sub_38814C0(v8, *(_QWORD *)(a1 + 56), (__int64)&v79);
                      }
                      else
                      {
                        v10 = sub_38A29E0(a1, (__int64)"thrownTypes", 11, (__int64)&v62, a4, a5, a6);
                      }
                    }
                    else
                    {
                      v10 = sub_38A29E0(a1, (__int64)"retainedNodes", 13, (__int64)&v60, a4, a5, a6);
                    }
                  }
                  else
                  {
                    v10 = sub_38A29E0(a1, (__int64)"declaration", 11, (__int64)&v58, a4, a5, a6);
                  }
                }
                else
                {
                  v10 = sub_38A29E0(a1, (__int64)"templateParams", 14, (__int64)&v56, a4, a5, a6);
                }
              }
              else
              {
                v10 = sub_38A29E0(a1, (__int64)"unit", 4, (__int64)&v54, a4, a5, a6);
              }
            }
            else
            {
              v10 = sub_3887760(a1, (__int64)"isOptimized", 11, (__int64)&v39);
            }
          }
          else
          {
            v10 = sub_388BBA0(a1, (__int64)"flags", 5, (__int64)&v40);
          }
        }
        else
        {
          v10 = sub_388AAB0(a1, (__int64)"thisAdjustment", 14, (__int64)v81);
        }
      }
      else
      {
        v10 = sub_3889510(a1, (__int64)"virtualIndex", 12, (__int64)v69);
      }
LABEL_6:
      if ( v10 )
        return 1;
      goto LABEL_7;
    }
LABEL_18:
    v25 = *(_QWORD *)(a1 + 56);
    v80 = 259;
    v79.m128i_i64[0] = (__int64)"expected field label here";
    if ( (unsigned __int8)sub_38814C0(v8, v25, (__int64)&v79) )
      return 1;
  }
LABEL_8:
  v11 = sub_388AF10(a1, 13, "expected ')' here");
  if ( (_BYTE)v11 )
    return 1;
  if ( (_BYTE)v38 )
  {
    if ( !a3 )
    {
      v80 = 259;
      v79.m128i_i64[0] = (__int64)"missing 'distinct', required for !DISubprogram when 'isDefinition'";
      return (unsigned int)sub_38814C0(v8, v7, (__int64)&v79);
    }
    v12 = v39;
    v13 = v40;
    v34 = v65[0];
    v14 = v81[0];
    v33 = v66;
    v15 = v69[0];
    v16 = v37;
    v17 = v64[0];
    v18 = v48;
    v19 = v46;
    v20 = v44;
    v21 = v42;
    goto LABEL_16;
  }
  v33 = v66;
  v12 = v39;
  v32 = *(_QWORD **)a1;
  v13 = v40;
  v34 = v65[0];
  v14 = v81[0];
  v15 = v69[0];
  v16 = v37;
  v17 = v64[0];
  v18 = v48;
  v19 = v46;
  v20 = v44;
  v21 = v42;
  if ( a3 )
  {
LABEL_16:
    v22 = sub_15BFC70(
            *(_QWORD **)a1,
            v21,
            v20,
            v19,
            v18,
            v17,
            v50,
            v16,
            v38,
            v34,
            v52,
            v33,
            v15,
            v14,
            v13,
            v12,
            v54,
            v56,
            v58,
            v60,
            v62,
            1u,
            1);
    goto LABEL_12;
  }
  v22 = sub_15BFC70(
          v32,
          v42,
          v44,
          v46,
          v48,
          v64[0],
          v50,
          v37,
          0,
          v65[0],
          v52,
          v66,
          v69[0],
          v81[0],
          v40,
          v39,
          v54,
          v56,
          v58,
          v60,
          v62,
          0,
          1);
LABEL_12:
  *a2 = v22;
  return v11;
}

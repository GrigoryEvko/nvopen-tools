// Function: sub_115F6E0
// Address: 0x115f6e0
//
unsigned __int8 *__fastcall sub_115F6E0(__m128i *a1, unsigned __int8 *a2)
{
  __int64 *v2; // r15
  const __m128i *v3; // r12
  __int64 v4; // rax
  char v5; // al
  __int64 v6; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 *v12; // r9
  __int64 v13; // rcx
  __int64 v14; // r8
  unsigned __int8 *v15; // r14
  __int64 v16; // rbx
  __int64 v17; // rcx
  __int64 v18; // r8
  unsigned __int8 *v19; // rbx
  __int64 v20; // rdx
  unsigned __int8 *v21; // rbx
  unsigned __int8 *v22; // rax
  __m128i *v23; // r8
  __m128i *v24; // rdi
  __int32 *v25; // rsi
  __int64 i; // rcx
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r9
  __m128i *v31; // rdi
  __int64 v32; // rcx
  const __m128i *v33; // rsi
  char v34; // al
  __int64 *v35; // rbx
  _BYTE *v36; // rax
  unsigned __int8 *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rdi
  unsigned __int8 *v41; // rax
  unsigned __int8 *v42; // rax
  __int64 v43; // rbx
  int v44; // eax
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  _BYTE *v49; // r14
  bool v50; // al
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rdi
  unsigned __int8 *v54; // rax
  __int64 v55; // r14
  __int64 v56; // rax
  unsigned int **v57; // r12
  _BYTE *v58; // r14
  __int64 v59; // rax
  __int64 v60; // r9
  __int64 v61; // rdx
  _BYTE *v62; // rax
  bool v63; // cl
  unsigned int v64; // r14d
  __int64 v65; // rsi
  __int64 v66; // rax
  bool v67; // [rsp+0h] [rbp-140h]
  int v68; // [rsp+0h] [rbp-140h]
  void *v69; // [rsp+10h] [rbp-130h]
  __int64 v70; // [rsp+10h] [rbp-130h]
  int v71; // [rsp+10h] [rbp-130h]
  __int64 *v72; // [rsp+18h] [rbp-128h]
  bool v73; // [rsp+18h] [rbp-128h]
  int v74; // [rsp+18h] [rbp-128h]
  __int64 v75; // [rsp+20h] [rbp-120h] BYREF
  __int64 *v76; // [rsp+28h] [rbp-118h] BYREF
  __int64 v77; // [rsp+30h] [rbp-110h] BYREF
  _BYTE *v78; // [rsp+38h] [rbp-108h] BYREF
  __m128i v79[2]; // [rsp+40h] [rbp-100h] BYREF
  __int64 *v80; // [rsp+68h] [rbp-D8h]
  __m128i v81; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v82; // [rsp+A0h] [rbp-A0h]
  __int64 **v83; // [rsp+B0h] [rbp-90h]
  void *v84; // [rsp+B8h] [rbp-88h]
  __m128i v85; // [rsp+C0h] [rbp-80h]
  __int64 v86; // [rsp+D0h] [rbp-70h]
  __int64 **v87; // [rsp+D8h] [rbp-68h]
  int v88; // [rsp+E0h] [rbp-60h]
  int v89; // [rsp+E8h] [rbp-58h]
  __int64 **v90; // [rsp+F0h] [rbp-50h]
  int v91; // [rsp+F8h] [rbp-48h]
  __int64 *v92; // [rsp+100h] [rbp-40h]

  v2 = (__int64 *)a2;
  v3 = a1;
  v4 = a1[10].m128i_i64[0];
  v81 = _mm_loadu_si128(a1 + 6);
  v83 = (__int64 **)_mm_loadu_si128(a1 + 8).m128i_u64[0];
  v82 = _mm_loadu_si128(a1 + 7);
  v84 = a2;
  v85 = _mm_loadu_si128(a1 + 9);
  v86 = v4;
  v5 = sub_B45210((__int64)a2);
  v6 = (__int64)sub_1009EB0(*((_BYTE **)a2 - 8), *((_BYTE **)a2 - 4), v5, &v81, 0, 1);
  if ( v6 )
  {
LABEL_2:
    a2 = (unsigned __int8 *)v2;
    a1 = (__m128i *)v3;
    return sub_F162A0((__int64)a1, (__int64)a2, v6);
  }
  if ( !(unsigned __int8)sub_F29CA0(a1, a2) )
  {
    v8 = (__int64)sub_F0F270((__int64)a1, a2);
    if ( v8 )
      return (unsigned __int8 *)v8;
    v8 = sub_F11DB0(a1->m128i_i64, a2);
    if ( v8 )
      return (unsigned __int8 *)v8;
    v8 = (__int64)sub_F28360((__int64)a1, a2, v9, v10, v11, v12);
    if ( v8 )
      return (unsigned __int8 *)v8;
    v6 = sub_115CCA0(a2, a1[2].m128i_i64[0]);
    if ( v6 )
      return sub_F162A0((__int64)a1, (__int64)a2, v6);
    v8 = (__int64)sub_115A080((__int64)a1, a2, 0, v13, v14);
    if ( v8 )
      return (unsigned __int8 *)v8;
    v8 = (__int64)sub_F18290(a1, a2);
    if ( v8 )
      return (unsigned __int8 *)v8;
    v15 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    v16 = *((_QWORD *)a2 - 8);
    v72 = (__int64 *)v16;
    v81.m128i_i64[0] = 0xBFF0000000000000LL;
    if ( (unsigned __int8)sub_1009690((double *)v81.m128i_i64, (__int64)v15) )
    {
      LOWORD(v83) = 257;
      v19 = (unsigned __int8 *)sub_B50340(12, v16, (__int64)&v81, 0, 0);
      sub_B45260(v19, (__int64)a2, 1);
      return v19;
    }
    v20 = *v15;
    v21 = v15 + 24;
    if ( (_BYTE)v20 != 18 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v15 + 1) + 8LL) - 17 > 1 )
        goto LABEL_30;
      if ( (unsigned __int8)v20 > 0x15u )
        goto LABEL_30;
      v36 = sub_AD7630((__int64)v15, 1, v20);
      if ( !v36 || *v36 != 18 )
        goto LABEL_30;
      v21 = v36 + 24;
    }
    v69 = sub_C33340();
    if ( *(void **)v21 == v69 )
      v22 = (unsigned __int8 *)*((_QWORD *)v21 + 1);
    else
      v22 = v21;
    if ( (v22[20] & 7) == 3 )
    {
      if ( sub_B451D0((__int64)a2) )
      {
        v31 = v79;
        v32 = 18;
        v33 = v3 + 6;
        while ( v32 )
        {
          v31->m128i_i32[0] = v33->m128i_i32[0];
          v33 = (const __m128i *)((char *)v33 + 4);
          v31 = (__m128i *)((char *)v31 + 4);
          --v32;
        }
        v80 = v2;
        v34 = sub_9B4030(v72, 3, 0, v79);
        v23 = (__m128i *)&v3[6];
        if ( (v34 & 3) == 0 )
          goto LABEL_25;
      }
      else
      {
        v23 = a1 + 6;
      }
      v24 = &v81;
      v25 = (__int32 *)v23;
      for ( i = 18; i; --i )
      {
        v24->m128i_i32[0] = *v25++;
        v24 = (__m128i *)((char *)v24 + 4);
      }
      v84 = v2;
      if ( (sub_9B4030(v2, 3, 0, &v81) & 3) == 0 )
      {
LABEL_25:
        if ( v69 == *(void **)v21 )
          v21 = (unsigned __int8 *)*((_QWORD *)v21 + 1);
        if ( (v21[20] & 8) != 0 )
        {
          v35 = (__int64 *)v3[2].m128i_i64[0];
          LOWORD(v83) = 257;
          sub_10A0170((__int64)v79, (__int64)v2);
          v72 = (__int64 *)sub_11553A0(v35, (__int64)v72, v79[0].m128i_u32[0], v79[0].m128i_i8[4], (__int64)&v81, 0);
        }
        v27 = v3[2].m128i_i64[0];
        LOWORD(v83) = 257;
        sub_10A0170((__int64)&v78, (__int64)v2);
        v79[0].m128i_i64[1] = (__int64)v72;
        v28 = v2[1];
        v79[0].m128i_i64[0] = (__int64)v15;
        v77 = v28;
        v6 = sub_B33D10(v27, 0x1Au, (__int64)&v77, 1, (int)v79, 2, (__int64)v78, (__int64)&v81);
        goto LABEL_2;
      }
    }
LABEL_30:
    v81.m128i_i64[0] = (__int64)&v75;
    if ( (unsigned __int8)sub_995E90(&v81, (unsigned __int64)v72, v20, v17, v18) )
    {
      if ( *v15 <= 0x15u )
      {
        v29 = sub_96E680(12, (__int64)v15);
        if ( v29 )
        {
          LOWORD(v83) = 257;
          return sub_109FE60(18, v75, v29, (__int64)v2, (__int64)&v81, v30, 0, 0);
        }
      }
    }
    if ( !sub_B451C0((__int64)v2) || !sub_B451E0((__int64)v2) )
    {
LABEL_47:
      v37 = sub_F0D870(v3, (unsigned __int8 *)v2, (__int64)v72, (__int64)v15);
      if ( v37 )
        return sub_F162A0((__int64)v3, (__int64)v2, (__int64)v37);
      if ( sub_B451B0((__int64)v2) )
      {
        v8 = (__int64)sub_115ACB0(v3, (__int64)v2, v45, v46, v47);
        if ( v8 )
          return (unsigned __int8 *)v8;
      }
      if ( !sub_B45190((__int64)v2) )
      {
LABEL_60:
        v77 = 0;
        v78 = 0;
        v79[0].m128i_i64[0] = 0;
        if ( sub_9913D0((__int64)v2, &v77, &v78, v79) && sub_B451C0((__int64)v2) )
        {
          v73 = sub_B451E0((__int64)v2);
          if ( v73 )
          {
            v49 = v78;
            if ( *v78 <= 0x15u )
            {
              if ( sub_AC30F0((__int64)v78) )
              {
LABEL_69:
                v37 = v78;
                return sub_F162A0((__int64)v3, (__int64)v2, (__int64)v37);
              }
              if ( *v49 == 17 )
              {
                if ( *((_DWORD *)v49 + 8) <= 0x40u )
                {
                  v50 = *((_QWORD *)v49 + 3) == 0;
                }
                else
                {
                  v74 = *((_DWORD *)v49 + 8);
                  v50 = v74 == (unsigned int)sub_C444A0((__int64)(v49 + 24));
                }
                if ( v50 )
                  goto LABEL_69;
              }
              else
              {
                v61 = *((_QWORD *)v49 + 1);
                v70 = v61;
                if ( (unsigned int)*(unsigned __int8 *)(v61 + 8) - 17 <= 1 )
                {
                  v62 = sub_AD7630((__int64)v49, 0, v61);
                  v63 = 0;
                  if ( v62 && *v62 == 17 )
                  {
                    v64 = *((_DWORD *)v62 + 8);
                    if ( v64 <= 0x40 )
                    {
                      if ( !*((_QWORD *)v62 + 3) )
                        goto LABEL_69;
                    }
                    else if ( v64 == (unsigned int)sub_C444A0((__int64)(v62 + 24)) )
                    {
                      goto LABEL_69;
                    }
                  }
                  else if ( *(_BYTE *)(v70 + 8) == 17 )
                  {
                    v65 = 0;
                    v71 = *(_DWORD *)(v70 + 32);
                    while ( v71 != (_DWORD)v65 )
                    {
                      v67 = v63;
                      v66 = sub_AD69F0(v49, v65);
                      if ( !v66 )
                        goto LABEL_63;
                      v63 = v67;
                      if ( *(_BYTE *)v66 != 13 )
                      {
                        if ( *(_BYTE *)v66 != 17 )
                          goto LABEL_63;
                        if ( *(_DWORD *)(v66 + 32) <= 0x40u )
                        {
                          if ( *(_QWORD *)(v66 + 24) )
                            goto LABEL_63;
                          v63 = v73;
                        }
                        else
                        {
                          v68 = *(_DWORD *)(v66 + 32);
                          if ( v68 != (unsigned int)sub_C444A0(v66 + 24) )
                            goto LABEL_63;
                          v63 = v73;
                        }
                      }
                      v65 = (unsigned int)(v65 + 1);
                    }
                    if ( v63 )
                      goto LABEL_69;
                  }
                }
              }
            }
          }
        }
LABEL_63:
        v85.m128i_i64[1] = (__int64)&v75;
        v82.m128i_i64[0] = (__int64)&v75;
        v81.m128i_i32[0] = 235;
        v81.m128i_i32[2] = 0;
        v82.m128i_i32[2] = 1;
        v83 = &v76;
        LODWORD(v84) = 246;
        v85.m128i_i32[0] = 0;
        LODWORD(v86) = 1;
        v87 = &v76;
        v88 = 246;
        v89 = 0;
        v90 = &v76;
        v91 = 1;
        v92 = &v75;
        if ( !sub_115F550((__int64)&v81, 18, (unsigned __int8 *)v2) )
          return 0;
        LOWORD(v83) = 257;
        v2 = (__int64 *)sub_109FE60(18, v75, (__int64)v76, (__int64)v2, (__int64)&v81, v48, 0, 0);
        if ( !sub_B451C0((__int64)v2) )
          sub_B44F10((__int64)v2, 0);
        return (unsigned __int8 *)v2;
      }
      v81.m128i_i32[0] = 220;
      v81.m128i_i32[2] = 0;
      v82.m128i_i64[0] = (__int64)&v75;
      v82.m128i_i64[1] = 0x3FE0000000000000LL;
      if ( (unsigned __int8)sub_11592A0((__int64)&v81, (__int64)v72) )
      {
        v76 = (__int64 *)v15;
        v81.m128i_i32[0] = 220;
        v81.m128i_i32[2] = 0;
        v82.m128i_i64[0] = (__int64)&v75;
        v82.m128i_i64[1] = 0x3FE0000000000000LL;
        if ( !(unsigned __int8)sub_11592A0((__int64)&v81, (__int64)v15) )
        {
          if ( !v72 )
            goto LABEL_60;
LABEL_82:
          v55 = v3[2].m128i_i64[0];
          LOWORD(v83) = 257;
          sub_10A0170((__int64)v79, (__int64)v2);
          v56 = sub_B33BC0(v55, 0xDCu, v75, v79[0].m128i_i64[0], (__int64)&v81);
          v57 = (unsigned int **)v3[2].m128i_i64[0];
          LOWORD(v83) = 257;
          v58 = (_BYTE *)v56;
          sub_10A0170((__int64)v79, (__int64)v2);
          v59 = sub_A826E0(v57, v58, v76, v79[0].m128i_i64[0], (__int64)&v81, 0);
          LOWORD(v83) = 257;
          return sub_109FE60(16, v59, (__int64)v76, (__int64)v2, (__int64)&v81, v60, 0, 0);
        }
      }
      else
      {
        v81.m128i_i32[0] = 220;
        v81.m128i_i32[2] = 0;
        v82.m128i_i64[0] = (__int64)&v75;
        v82.m128i_i64[1] = 0x3FE0000000000000LL;
        if ( !(unsigned __int8)sub_11592A0((__int64)&v81, (__int64)v15) )
          goto LABEL_60;
      }
      v76 = v72;
      goto LABEL_82;
    }
    if ( *(_BYTE *)v72 == 72
      && (v51 = *(v72 - 4)) != 0
      && (v52 = *(_QWORD *)(v51 + 8), v75 = *(v72 - 4), sub_1001970(v52, 1)) )
    {
      v53 = v2[1];
      LOWORD(v83) = 257;
      v54 = sub_AD8DD0(v53, 0.0);
      v42 = sub_109FEA0(v75, (__int64)v15, (__int64)v54, (const char **)&v81, 0, 0, 0);
    }
    else
    {
      if ( *v15 != 72 )
        goto LABEL_47;
      v38 = *((_QWORD *)v15 - 4);
      if ( !v38 )
        goto LABEL_47;
      v39 = *(_QWORD *)(v38 + 8);
      v75 = *((_QWORD *)v15 - 4);
      if ( !sub_1001970(v39, 1) )
        goto LABEL_47;
      v40 = v2[1];
      LOWORD(v83) = 257;
      v41 = sub_AD8DD0(v40, 0.0);
      v42 = sub_109FEA0(v75, (__int64)v72, (__int64)v41, (const char **)&v81, 0, 0, 0);
    }
    v43 = (__int64)v42;
    v44 = sub_B45210((__int64)v2);
    v2 = (__int64 *)v43;
    sub_B45170(v43, v44);
  }
  return (unsigned __int8 *)v2;
}

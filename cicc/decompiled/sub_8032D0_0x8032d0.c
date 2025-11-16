// Function: sub_8032D0
// Address: 0x8032d0
//
_QWORD *__fastcall sub_8032D0(
        const __m128i *a1,
        __int64 a2,
        int a3,
        __int64 *a4,
        int a5,
        __m128i *a6,
        _DWORD *a7,
        int a8)
{
  _QWORD *v8; // r14
  __int64 i; // r15
  __m128i v11; // xmm2
  __int64 v12; // rdx
  char v13; // dl
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // r14
  __int8 v17; // al
  __m128i *v18; // rbx
  int v19; // r12d
  int v20; // r8d
  char v21; // dl
  __int64 *v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rax
  const __m128i *v28; // rax
  _QWORD *result; // rax
  __int64 v30; // rdx
  __int64 v31; // r15
  _BOOL4 v32; // eax
  __int64 v33; // rcx
  __int64 v34; // r8
  int v35; // eax
  __int8 v36; // al
  __int64 v37; // rax
  __m128i *v38; // rax
  __m128i *v39; // rax
  __int64 *v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rdx
  int v43; // eax
  int v44; // r8d
  int v45; // eax
  __m128i *v46; // rax
  __int64 v47; // rdi
  __m128i *v48; // r15
  _QWORD *v49; // rax
  int v50; // r8d
  __int64 j; // r15
  _QWORD *v52; // rax
  __int64 v53; // rcx
  _QWORD *v54; // r12
  __m128i *v55; // r11
  __int8 v56; // al
  __int64 v57; // r12
  _QWORD *v58; // rax
  __m128i *v59; // r12
  _BYTE *v60; // rax
  _BYTE *v61; // rax
  _BYTE *v62; // rax
  __int64 v63; // rdi
  _BYTE *v64; // rax
  _QWORD *v65; // rax
  _QWORD *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  int v71; // eax
  __int64 v72; // rax
  __int64 v73; // rax
  _QWORD *v74; // rax
  __int64 v75; // [rsp-8h] [rbp-298h]
  __int64 v76; // [rsp+8h] [rbp-288h]
  __m128i *v77; // [rsp+10h] [rbp-280h]
  _QWORD *v78; // [rsp+18h] [rbp-278h]
  _BYTE *v79; // [rsp+18h] [rbp-278h]
  _QWORD *v80; // [rsp+20h] [rbp-270h]
  int v81; // [rsp+28h] [rbp-268h]
  __int64 v82; // [rsp+28h] [rbp-268h]
  __int64 v83; // [rsp+28h] [rbp-268h]
  __int64 *v84; // [rsp+28h] [rbp-268h]
  _QWORD *v85; // [rsp+30h] [rbp-260h]
  int v86; // [rsp+38h] [rbp-258h]
  int v87; // [rsp+38h] [rbp-258h]
  __m128i *v88; // [rsp+38h] [rbp-258h]
  int v89; // [rsp+40h] [rbp-250h]
  __m128i *v92; // [rsp+50h] [rbp-240h]
  __int64 v94; // [rsp+60h] [rbp-230h]
  __m128i *v95; // [rsp+68h] [rbp-228h]
  bool v97; // [rsp+7Bh] [rbp-215h]
  unsigned int v98; // [rsp+7Ch] [rbp-214h]
  unsigned int v99; // [rsp+8Ch] [rbp-204h] BYREF
  _BYTE v100[32]; // [rsp+90h] [rbp-200h] BYREF
  __m128i v101[2]; // [rsp+B0h] [rbp-1E0h] BYREF
  __int64 v102; // [rsp+D0h] [rbp-1C0h] BYREF
  __int64 v103; // [rsp+D8h] [rbp-1B8h]
  __int64 v104; // [rsp+E0h] [rbp-1B0h]
  __int64 v105; // [rsp+E8h] [rbp-1A8h]
  __int64 v106; // [rsp+F0h] [rbp-1A0h]
  char v107; // [rsp+F8h] [rbp-198h]
  __m128i v108; // [rsp+100h] [rbp-190h] BYREF
  __m128i v109; // [rsp+110h] [rbp-180h]
  __m128i v110; // [rsp+120h] [rbp-170h]
  __m128i v111; // [rsp+130h] [rbp-160h]
  __int64 v112; // [rsp+140h] [rbp-150h]
  __m128i v113[5]; // [rsp+150h] [rbp-140h] BYREF
  _BYTE v114[240]; // [rsp+1A0h] [rbp-F0h] BYREF

  v8 = (_QWORD *)a2;
  a1[-1].m128i_i8[8] |= 8u;
  for ( i = sub_7F9140(a2); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (unsigned int)sub_8D2B50(i) )
  {
    i = sub_7D7990(*(_BYTE *)(i + 160));
    sub_7D8C20(a1);
  }
  v11 = _mm_loadu_si128((const __m128i *)(a2 + 32));
  v112 = *(_QWORD *)(a2 + 64);
  v110 = v11;
  v12 = v11.m128i_i64[0];
  v108 = _mm_loadu_si128((const __m128i *)a2);
  v109 = _mm_loadu_si128((const __m128i *)(a2 + 16));
  v111 = _mm_loadu_si128((const __m128i *)(a2 + 48));
  if ( dword_4F077C4 == 2 )
  {
    if ( v11.m128i_i64[0] )
    {
      v41 = *(_QWORD *)(v11.m128i_i64[0] + 24);
      if ( v41 )
      {
        if ( (*(_BYTE *)(v41 + 144) & 0x10) != 0 )
          v12 = *(_QWORD *)v11.m128i_i64[0];
      }
    }
  }
  v102 = v12;
  v110.m128i_i64[0] = (__int64)&v102;
  v13 = *(_BYTE *)(i + 140);
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  if ( v13 == 8 )
  {
    v94 = 0;
    v98 = 1;
    v103 = *(_QWORD *)(i + 160);
  }
  else
  {
    if ( v13 != 15 )
    {
      sub_7E31E0((__int64)a1);
      a2 = 11;
      v94 = sub_72FD90(*(_QWORD *)(i + 160), 11);
      if ( qword_4D03EA8 )
        *v8 = qword_4D03EA8;
      qword_4D03EA8 = v8;
      v98 = 0;
      if ( dword_4F077C4 != 2 )
        goto LABEL_10;
      goto LABEL_51;
    }
    v14 = *(_QWORD *)(i + 160);
    v107 = 1;
    v94 = 0;
    v103 = v14;
    v98 = 1;
  }
  if ( dword_4F077C4 == 2 )
  {
LABEL_51:
    if ( unk_4F07778 > 201102 || dword_4F07774 )
      a8 &= ~1u;
  }
LABEL_10:
  v15 = a1[11].m128i_i64[0];
  if ( !v15 )
    goto LABEL_43;
  v95 = 0;
  v85 = v8;
  v16 = i;
  v97 = a5 != 0;
  do
  {
    v17 = *(_BYTE *)(v15 + 173);
    v18 = (__m128i *)v15;
    if ( v17 == 13 )
    {
      if ( (*(_BYTE *)(v15 + 176) & 1) != 0 )
        v94 = *(_QWORD *)(v15 + 184);
      else
        v104 = *(_QWORD *)(v15 + 184);
      v18 = *(__m128i **)(v15 + 120);
      v17 = v18[10].m128i_i8[13];
    }
    v15 = v18[7].m128i_i64[1];
    v19 = 0;
    v20 = v97 || v15 != 0;
    if ( v98 )
      goto LABEL_20;
    v21 = v18[10].m128i_i8[11];
    v109.m128i_i8[2] = 0;
    v106 = 0;
    v105 = 0;
    if ( (v21 & 0x20) != 0 )
    {
      a2 = (__int64)&v18[12].m128i_i64[1];
      v22 = &v18[11].m128i_i64[1];
      if ( v17 == 10 )
        v22 = &v18[12].m128i_i64[1];
      v23 = *v22;
      if ( v21 < 0 )
      {
        v24 = *(_QWORD *)(v23 + 40);
        v106 = v23;
        v19 = 1;
        v109.m128i_i8[2] = 1;
      }
      else
      {
        v24 = *(_QWORD *)(v23 + 120);
        v105 = v23;
        v19 = 1;
      }
      v103 = v24;
LABEL_20:
      if ( v17 == 9 )
        goto LABEL_60;
      goto LABEL_21;
    }
    v30 = *(_QWORD *)(v94 + 120);
    v105 = v94;
    v103 = v30;
    if ( v17 == 9 )
    {
LABEL_60:
      a2 = (__int64)&v108;
      sub_802FE0(v18, &v108, a3, a4, v20, a6, a7, a8);
      goto LABEL_29;
    }
LABEL_21:
    if ( v17 == 11 )
    {
      v92 = (__m128i *)v18[11].m128i_i64[0];
      v31 = sub_8D4050(v16);
      v32 = sub_7F50D0((__int64)v92);
      v34 = v97 | (unsigned __int8)(v15 != 0);
      if ( v32 )
      {
        a2 = *(_QWORD *)(v33 + 128);
        if ( a2 != v31 )
        {
          v35 = sub_8D97D0(v31, a2, 32, v33, v34);
          LODWORD(v34) = v97 || v15 != 0;
          if ( !v35 )
          {
            a2 = v31;
            sub_7F7440((__int64)v18, v31);
            LODWORD(v34) = v97 || v15 != 0;
            v92 = (__m128i *)v18[11].m128i_i64[0];
          }
        }
      }
      v36 = v92[10].m128i_i8[13];
      if ( v36 == 9 || v36 == 10 && (v92[12].m128i_i8[0] & 1) != 0 )
      {
        v109.m128i_i8[1] = 1;
        v111.m128i_i64[0] = v92[8].m128i_i64[0];
        v42 = v18[11].m128i_i64[1];
        if ( v42 )
        {
          if ( *((_BYTE *)v85 + 17) )
            v42 *= v85[5];
          v110.m128i_i64[1] = v42;
        }
        else
        {
          v86 = v34;
          v43 = sub_8D4070(v16);
          v44 = v86;
          if ( v43 )
          {
            v74 = sub_7D78E0(v16);
            v44 = v86;
            v111.m128i_i64[1] = (__int64)v74;
          }
          v87 = v44;
          v112 = v104;
          v45 = sub_8DD010(v16);
          LODWORD(v34) = v87;
          if ( v45 || (v70 = sub_8D4050(v16), v71 = sub_8D3410(v70), LODWORD(v34) = v87, !v71) )
          {
            v36 = v92[10].m128i_i8[13];
          }
          else
          {
            v72 = sub_8D4050(v16);
            v73 = sub_8D4490(v72);
            LODWORD(v34) = v87;
            v112 *= v73;
            v36 = v92[10].m128i_i8[13];
          }
        }
        if ( v36 != 9 || *(_BYTE *)(v92[11].m128i_i64[0] + 48) == 6 )
        {
          v81 = v34;
          v46 = (__m128i *)sub_7F98A0((__int64)&v108, 0);
          v47 = v18[11].m128i_i64[1];
          v88 = v46;
          v48 = v46;
          if ( v47 )
          {
            v66 = sub_73A8E0(v47, byte_4F06A51[0]);
            v50 = v81;
            v88[1].m128i_i64[0] = (__int64)v66;
          }
          else
          {
            v49 = sub_7E8090((const __m128i *)v111.m128i_i64[1], 1u);
            v50 = v81;
            v48[1].m128i_i64[0] = (__int64)v49;
          }
          for ( j = v88->m128i_i64[0]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          v89 = v50;
          v80 = sub_72BA30(byte_4F06A51[0]);
          v82 = sub_72CBE0();
          v52 = sub_7259C0(7);
          v53 = v52[21];
          v54 = v52;
          v52[20] = v82;
          *(_BYTE *)(v53 + 16) = (2 * (dword_4F06968 == 0)) | *(_BYTE *)(v53 + 16) & 0xFD;
          *(_QWORD *)v52[21] = sub_724EF0(j);
          v55 = sub_725FD0();
          v55[10].m128i_i8[12] = 2;
          v56 = v55[5].m128i_i8[8];
          v55[12].m128i_i8[1] |= 0x10u;
          v55[9].m128i_i64[1] = (__int64)v54;
          v83 = (__int64)v55;
          v55[5].m128i_i8[8] = v56 & 0x8F | 0x10;
          sub_7362F0((__int64)v55, 0);
          v57 = *(_QWORD *)(*(_QWORD *)(v83 + 152) + 168LL);
          v58 = sub_724EF0((__int64)v80);
          v76 = v83;
          **(_QWORD **)v57 = v58;
          v78 = v58;
          v84 = sub_7F54F0(v83, 1, dword_4F07270[0], &v99);
          sub_7F6C60((__int64)v84, v99, (__int64)v114);
          v59 = sub_7E2270(*(_QWORD *)(*(_QWORD *)v57 + 8LL));
          v77 = sub_7E2270(v78[1]);
          v84[5] = (__int64)v59;
          v59[7].m128i_i64[0] = (__int64)v77;
          sub_7E1740(v84[10], (__int64)v100);
          sub_7E2BA0((__int64)v100);
          sub_7FAFA0((__int64)v100);
          v79 = sub_726B30(5);
          sub_7E6810((__int64)v79, (__int64)v100, 1);
          v60 = sub_731250((__int64)v77);
          v61 = sub_73DBF0(0x24u, (__int64)v80, (__int64)v60);
          *((_QWORD *)v79 + 6) = sub_7F0830(v61);
          v62 = sub_726B30(11);
          *((_QWORD *)v79 + 9) = v62;
          sub_7E1740((__int64)v62, (__int64)v101);
          sub_7F90D0((__int64)v59, (__int64)v113);
          if ( v92[10].m128i_i8[13] == 10 )
            sub_8032D0((_DWORD)v92, (unsigned int)v113, a3, (_DWORD)a4, v89, (unsigned int)v101, (__int64)a7, a8);
          else
            sub_802FE0(v92, v113, a3, a4, v89, v101, a7, a8);
          v63 = (__int64)v59;
          v19 = 1;
          v64 = sub_731250(v63);
          v65 = sub_73DBF0(0x23u, j, (__int64)v64);
          sub_7E69E0(v65, v101[0].m128i_i32);
          sub_7FB010((__int64)v84, v99, (__int64)v114);
          a2 = (__int64)v88;
          sub_7F88F0(v76, v88, 0, a6);
        }
        else
        {
          a2 = (__int64)&v108;
          sub_802FE0(v92, &v108, a3, a4, v34, a6, a7, a8);
        }
      }
      else
      {
        if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 || dword_4F077C0 )
          sub_7D8CF0(v92);
        else
          sub_7EB190((__int64)v92, (__m128i *)a2);
        if ( !v109.m128i_i8[0] )
          goto LABEL_28;
        v109.m128i_i8[1] = 1;
        v111.m128i_i64[0] = v92[8].m128i_i64[0];
        v37 = v18[11].m128i_i64[1];
        if ( v37 )
        {
          if ( *((_BYTE *)v85 + 17) )
            v37 *= v85[5];
          v110.m128i_i64[1] = v37;
        }
        else
        {
          if ( (unsigned int)sub_8D4070(v16) )
            v111.m128i_i64[1] = (__int64)sub_7D78E0(v16);
          v112 = v104;
          v67 = sub_8D4050(v16);
          if ( (unsigned int)sub_8D3410(v67) )
          {
            v68 = sub_8D4050(v16);
            v69 = sub_8D4490(v68);
            v112 *= v69;
          }
        }
        v38 = sub_7F9430((__int64)&v108, 1, 1);
        v92[7].m128i_i64[1] = 0;
        a2 = (__int64)v92;
        sub_7FBC20(0, (__int64)v92, v38->m128i_i64, 0, a6, 0, (__int64)&v108);
      }
    }
    else if ( v17 == 10 )
    {
      if ( v18[11].m128i_i64[0] )
      {
        a2 = (__int64)&v108;
        sub_8032D0((_DWORD)v18, (unsigned int)&v108, a3, (_DWORD)a4, v20, (_DWORD)a6, (__int64)a7, a8);
      }
      else
      {
        v18[-1].m128i_i8[8] |= 8u;
        if ( !v109.m128i_i8[0] )
          goto LABEL_28;
        v40 = (__int64 *)sub_7F98A0((__int64)&v108, 1);
        a2 = 1;
        sub_7FB7C0(v18[8].m128i_i64[0], 1u, v40, 0, 0, 0, a6);
      }
    }
    else
    {
      if ( dword_4F077C4 == 2 )
      {
        sub_7EB190((__int64)v18, (__m128i *)a2);
      }
      else if ( unk_4F07778 > 199900 || dword_4F077C0 )
      {
        sub_7D8CF0(v18);
      }
      if ( !v109.m128i_i8[0] )
      {
LABEL_28:
        *a7 = 1;
        goto LABEL_29;
      }
      v39 = sub_7F9430((__int64)&v108, 1, 1);
      v18[7].m128i_i64[1] = 0;
      sub_7FBC20(0, (__int64)v18, v39->m128i_i64, 0, a6, 0, (__int64)&v108);
      a2 = v75;
    }
LABEL_29:
    if ( v98 )
    {
      if ( v18[10].m128i_i8[13] == 11 )
        v25 = v18[11].m128i_i64[1] + v104;
      else
        v25 = v104 + 1;
      v104 = v25;
    }
    else if ( (v18[10].m128i_i8[11] & 0x20) == 0 )
    {
      a2 = 11;
      v94 = sub_72FD90(*(_QWORD *)(v94 + 112), 11);
    }
    if ( (unsigned __int8)(*(_BYTE *)(v16 + 140) - 9) <= 2u && (*(_BYTE *)(*(_QWORD *)(v16 + 168) + 109LL) & 0x20) != 0 )
    {
      a2 = (__int64)a4;
      if ( a4 )
      {
        v26 = (__int64 *)a4[1];
        if ( v26 )
          a4[1] = *v26;
      }
    }
    if ( !v19 )
    {
      v95 = v18;
      continue;
    }
    v27 = v18[7].m128i_i64[1];
    if ( v95 )
    {
      a2 = (__int64)v95;
      v95[7].m128i_i64[1] = v27;
      v28 = a1;
      if ( (__m128i *)a1[11].m128i_i64[1] != v18 )
        continue;
    }
    else
    {
      a2 = (__int64)a1;
      a1[11].m128i_i64[0] = v27;
      v28 = a1;
      if ( (__m128i *)a1[11].m128i_i64[1] != v18 )
        continue;
    }
    v28[11].m128i_i64[1] = (__int64)v95;
  }
  while ( v15 );
  i = v16;
LABEL_43:
  if ( (a1[10].m128i_i8[10] & 0x40) == 0 )
    sub_7F7570((__int64)a1, i);
  result = (_QWORD *)v98;
  if ( !v98 )
  {
    result = qword_4D03EA8;
    qword_4D03EA8 = (_QWORD *)*qword_4D03EA8;
    *result = 0;
  }
  return result;
}

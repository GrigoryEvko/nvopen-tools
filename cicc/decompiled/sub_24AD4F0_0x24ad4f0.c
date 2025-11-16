// Function: sub_24AD4F0
// Address: 0x24ad4f0
//
void __fastcall sub_24AD4F0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v5; // r9
  _QWORD *v6; // r10
  unsigned __int64 v7; // r14
  _QWORD *v9; // r15
  _QWORD *v10; // r13
  __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned int *v13; // rdx
  unsigned int *v14; // rsi
  __int64 v15; // rdx
  unsigned int *v16; // rdi
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rcx
  _QWORD *v20; // r10
  __int64 v21; // r9
  unsigned int v22; // r13d
  int v23; // eax
  int v24; // eax
  int v25; // eax
  _QWORD *v26; // r10
  unsigned __int64 v27; // rsi
  unsigned int *v28; // rdx
  unsigned int *v29; // rcx
  __int64 v30; // rax
  unsigned __int64 v31; // r8
  unsigned __int64 v32; // r9
  unsigned __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __m128i v45; // xmm0
  __m128i v46; // xmm1
  __m128i v47; // xmm2
  unsigned __int64 *v48; // r14
  unsigned __int64 *v49; // rbx
  unsigned __int64 *v50; // r15
  unsigned __int64 v51; // rdi
  __int64 *v52; // r14
  unsigned __int64 *v53; // rbx
  __int64 v54; // r8
  unsigned __int64 v55; // rdi
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rsi
  __int64 v59; // [rsp+0h] [rbp-490h]
  _QWORD *v60; // [rsp+8h] [rbp-488h]
  __int64 v61; // [rsp+10h] [rbp-480h]
  _QWORD *v62; // [rsp+18h] [rbp-478h]
  __int64 v63; // [rsp+18h] [rbp-478h]
  __int64 v64; // [rsp+18h] [rbp-478h]
  __int64 v65; // [rsp+18h] [rbp-478h]
  __int64 v66; // [rsp+18h] [rbp-478h]
  __int64 v67; // [rsp+20h] [rbp-470h]
  int v68; // [rsp+20h] [rbp-470h]
  __int64 v69; // [rsp+20h] [rbp-470h]
  _QWORD *v70; // [rsp+20h] [rbp-470h]
  _QWORD *v71; // [rsp+20h] [rbp-470h]
  unsigned __int64 v72; // [rsp+20h] [rbp-470h]
  __int64 v73; // [rsp+20h] [rbp-470h]
  _QWORD *v74; // [rsp+20h] [rbp-470h]
  _QWORD *v75; // [rsp+20h] [rbp-470h]
  _QWORD *v76; // [rsp+20h] [rbp-470h]
  _QWORD *v77; // [rsp+28h] [rbp-468h]
  int v78; // [rsp+3Ch] [rbp-454h] BYREF
  __int64 v79[2]; // [rsp+40h] [rbp-450h] BYREF
  __int64 *v80; // [rsp+50h] [rbp-440h]
  unsigned int *v81; // [rsp+60h] [rbp-430h] BYREF
  __int64 v82; // [rsp+68h] [rbp-428h]
  _BYTE v83[16]; // [rsp+70h] [rbp-420h] BYREF
  __m128i *v84; // [rsp+80h] [rbp-410h] BYREF
  size_t v85; // [rsp+88h] [rbp-408h]
  __m128i si128; // [rsp+90h] [rbp-400h] BYREF
  __int8 *v87; // [rsp+A0h] [rbp-3F0h] BYREF
  size_t v88; // [rsp+A8h] [rbp-3E8h]
  _BYTE v89[16]; // [rsp+B0h] [rbp-3E0h] BYREF
  _QWORD v90[8]; // [rsp+C0h] [rbp-3D0h] BYREF
  __int8 *v91; // [rsp+100h] [rbp-390h] BYREF
  size_t v92; // [rsp+108h] [rbp-388h]
  _BYTE v93[24]; // [rsp+110h] [rbp-380h] BYREF
  __int64 v94; // [rsp+128h] [rbp-368h]
  __m128i v95; // [rsp+130h] [rbp-360h]
  __m128i v96; // [rsp+140h] [rbp-350h]
  unsigned __int64 *v97; // [rsp+150h] [rbp-340h] BYREF
  __int64 v98; // [rsp+158h] [rbp-338h]
  _BYTE v99[320]; // [rsp+160h] [rbp-330h] BYREF
  char v100; // [rsp+2A0h] [rbp-1F0h]
  int v101; // [rsp+2A4h] [rbp-1ECh]
  __int64 v102; // [rsp+2A8h] [rbp-1E8h]
  void *v103; // [rsp+2B0h] [rbp-1E0h] BYREF
  __int64 v104; // [rsp+2B8h] [rbp-1D8h]
  __int64 v105; // [rsp+2C0h] [rbp-1D0h]
  __m128i v106; // [rsp+2C8h] [rbp-1C8h] BYREF
  __int64 v107; // [rsp+2D8h] [rbp-1B8h]
  __m128i v108; // [rsp+2E0h] [rbp-1B0h] BYREF
  __m128i v109; // [rsp+2F0h] [rbp-1A0h] BYREF
  unsigned __int64 *v110; // [rsp+300h] [rbp-190h] BYREF
  unsigned int v111; // [rsp+308h] [rbp-188h]
  char v112; // [rsp+310h] [rbp-180h] BYREF
  char v113; // [rsp+450h] [rbp-40h]
  int v114; // [rsp+454h] [rbp-3Ch]
  __int64 v115; // [rsp+458h] [rbp-38h]

  v5 = (__int64)a3;
  v6 = a3;
  v7 = 1;
  if ( a5 > 0xFFFFFFFE )
    v7 = a5 / 0xFFFFFFFF + 1;
  v9 = &a3[a4];
  v81 = (unsigned int *)v83;
  v82 = 0x400000000LL;
  if ( v9 == a3 )
  {
    v14 = (unsigned int *)v83;
    v15 = 0;
  }
  else
  {
    v10 = a3 + 1;
    v11 = 0;
    v12 = *a3 / v7;
    v13 = (unsigned int *)v83;
    while ( 1 )
    {
      v13[v11] = v12;
      v11 = (unsigned int)(v82 + 1);
      LODWORD(v82) = v82 + 1;
      if ( v9 == v10 )
        break;
      v12 = *v10 / v7;
      if ( v11 + 1 > (unsigned __int64)HIDWORD(v82) )
      {
        v61 = v5;
        v62 = v6;
        v68 = *v10 / v7;
        sub_C8D5F0((__int64)&v81, v83, v11 + 1, 4u, a5, v5);
        v11 = (unsigned int)v82;
        v5 = v61;
        v6 = v62;
        LODWORD(v12) = v68;
      }
      v13 = v81;
      ++v10;
    }
    v14 = v81;
    v15 = (unsigned int)v11;
  }
  v67 = v5;
  v77 = v6;
  sub_2A3E730(a2, v14, v15, 0);
  sub_BC8EC0(a2, v81, (unsigned int)v82, 0);
  if ( !byte_4FEB868 )
  {
LABEL_11:
    v16 = v81;
    if ( v81 == (unsigned int *)v83 )
      return;
    goto LABEL_12;
  }
  if ( *(_BYTE *)a2 != 31 || (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 3 || (v17 = *(_QWORD *)(a2 - 96), *(_BYTE *)v17 != 82) )
  {
    si128.m128i_i8[0] = 0;
    v84 = &si128;
    v85 = 0;
    goto LABEL_18;
  }
  v91 = v93;
  v107 = 0x100000000LL;
  v108.m128i_i64[0] = (__int64)&v91;
  v103 = &unk_49DD210;
  v92 = 0;
  v93[0] = 0;
  v104 = 0;
  v105 = 0;
  v106 = 0u;
  sub_CB5980((__int64)&v103, 0, 0, 0);
  v18 = sub_B52E10((__int64)&v103, *(_WORD *)(v17 + 2) & 0x3F);
  sub_904010(v18, "_");
  sub_A587F0(*(_QWORD *)(*(_QWORD *)(v17 - 64) + 8LL), (__int64)&v103, 1, 0);
  v19 = *(_QWORD *)(v17 - 32);
  v20 = v77;
  v21 = v67;
  if ( *(_BYTE *)v19 == 17 )
  {
    v22 = *(_DWORD *)(v19 + 32);
    if ( v22 <= 0x40 )
    {
      if ( *(_QWORD *)(v19 + 24) )
      {
        v58 = *(_QWORD *)(v19 + 24);
        if ( v58 != 1 )
        {
          if ( v22 && v58 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22) )
            goto LABEL_26;
LABEL_75:
          v65 = v21;
          v75 = v20;
          sub_904010((__int64)&v103, "_MinusOne");
          v20 = v75;
          v21 = v65;
          goto LABEL_27;
        }
        goto LABEL_78;
      }
    }
    else
    {
      v59 = v67;
      v69 = v19 + 24;
      v23 = sub_C444A0(v19 + 24);
      v20 = v77;
      v21 = v59;
      if ( v22 != v23 )
      {
        v24 = sub_C444A0(v69);
        v20 = v77;
        v21 = v59;
        if ( v24 != v22 - 1 )
        {
          v25 = sub_C445E0(v69);
          v20 = v77;
          v21 = v59;
          if ( v22 != v25 )
          {
LABEL_26:
            v63 = v21;
            v70 = v20;
            sub_904010((__int64)&v103, "_Const");
            v21 = v63;
            v20 = v70;
            goto LABEL_27;
          }
          goto LABEL_75;
        }
LABEL_78:
        v66 = v21;
        v76 = v20;
        sub_904010((__int64)&v103, "_One");
        v20 = v76;
        v21 = v66;
        goto LABEL_27;
      }
    }
    v64 = v21;
    v74 = v20;
    sub_904010((__int64)&v103, "_Zero");
    v20 = v74;
    v21 = v64;
  }
LABEL_27:
  v84 = &si128;
  if ( v91 == v93 )
  {
    si128 = _mm_load_si128((const __m128i *)v93);
  }
  else
  {
    v84 = (__m128i *)v91;
    si128.m128i_i64[0] = *(_QWORD *)v93;
  }
  v60 = (_QWORD *)v21;
  v71 = v20;
  v85 = v92;
  v92 = 0;
  v91 = v93;
  v93[0] = 0;
  v103 = &unk_49DD210;
  sub_CB5840((__int64)&v103);
  sub_2240A30((unsigned __int64 *)&v91);
  v26 = v71;
  if ( v85 )
  {
    v27 = 0;
    v28 = &v81[(unsigned int)v82];
    v29 = v81;
    if ( v81 == v28 )
    {
      if ( v9 == v60 )
      {
        LODWORD(v32) = 0;
        v31 = 0;
        v33 = 1;
LABEL_37:
        v72 = v31;
        sub_F02DB0(&v78, *v81 / v33, v32);
        v87 = v89;
        v90[5] = 0x100000000LL;
        v90[6] = &v87;
        v88 = 0;
        v90[0] = &unk_49DD210;
        v89[0] = 0;
        memset(&v90[1], 0, 32);
        sub_CB5980((__int64)v90, 0, 0, 0);
        LODWORD(v103) = v78;
        sub_F02CC0((int *)&v103, (__int64)v90, v34, v35, v36, v37);
        v38 = sub_904010((__int64)v90, " (total count : ");
        v39 = sub_CB59D0(v38, v72);
        sub_904010(v39, ")");
        sub_1049690(v79, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL));
        v73 = v79[0];
        v40 = sub_B2BE50(v79[0]);
        if ( sub_B6EA50(v40)
          || (v56 = sub_B2BE50(v73),
              v57 = sub_B6F970(v56),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v57 + 48LL))(v57)) )
        {
          sub_B174A0((__int64)&v103, (__int64)"pgo-instrumentation", (__int64)"pgo-instrumentation", 19, a2);
          sub_B18290((__int64)&v103, v84->m128i_i8, v85);
          sub_B18290((__int64)&v103, " is true with probability : ", 0x1Cu);
          sub_B18290((__int64)&v103, v87, v88);
          v45 = _mm_loadu_si128(&v106);
          v46 = _mm_load_si128(&v108);
          v47 = _mm_load_si128(&v109);
          LODWORD(v92) = v104;
          *(__m128i *)&v93[8] = v45;
          BYTE4(v92) = BYTE4(v104);
          v95 = v46;
          *(_QWORD *)v93 = v105;
          v96 = v47;
          v91 = (__int8 *)&unk_49D9D40;
          v94 = v107;
          v97 = (unsigned __int64 *)v99;
          v98 = 0x400000000LL;
          if ( v111 )
          {
            sub_24AC760((__int64)&v97, (__int64)&v110, v41, v42, v43, v44);
            v103 = &unk_49D9D40;
            v53 = v110;
            v100 = v113;
            v101 = v114;
            v102 = v115;
            v91 = (__int8 *)&unk_49D9D78;
            v54 = 10LL * v111;
            v48 = &v110[v54];
            if ( v110 != &v110[v54] )
            {
              do
              {
                v48 -= 10;
                v55 = v48[4];
                if ( (unsigned __int64 *)v55 != v48 + 6 )
                  j_j___libc_free_0(v55);
                if ( (unsigned __int64 *)*v48 != v48 + 2 )
                  j_j___libc_free_0(*v48);
              }
              while ( v53 != v48 );
              v48 = v110;
            }
          }
          else
          {
            v48 = v110;
            v100 = v113;
            v101 = v114;
            v102 = v115;
            v91 = (__int8 *)&unk_49D9D78;
          }
          if ( v48 != (unsigned __int64 *)&v112 )
            _libc_free((unsigned __int64)v48);
          sub_1049740(v79, (__int64)&v91);
          v49 = v97;
          v91 = (__int8 *)&unk_49D9D40;
          v50 = &v97[10 * (unsigned int)v98];
          if ( v97 != v50 )
          {
            do
            {
              v50 -= 10;
              v51 = v50[4];
              if ( (unsigned __int64 *)v51 != v50 + 6 )
                j_j___libc_free_0(v51);
              if ( (unsigned __int64 *)*v50 != v50 + 2 )
                j_j___libc_free_0(*v50);
            }
            while ( v49 != v50 );
            v50 = v97;
          }
          if ( v50 != (unsigned __int64 *)v99 )
            _libc_free((unsigned __int64)v50);
        }
        v52 = v80;
        if ( v80 )
        {
          sub_FDC110(v80);
          j_j___libc_free_0((unsigned __int64)v52);
        }
        v90[0] = &unk_49DD210;
        sub_CB5840((__int64)v90);
        if ( v87 != v89 )
          j_j___libc_free_0((unsigned __int64)v87);
        if ( v84 != &si128 )
          j_j___libc_free_0((unsigned __int64)v84);
        goto LABEL_11;
      }
    }
    else
    {
      do
      {
        v30 = *v29++;
        v27 += v30;
      }
      while ( v29 != v28 );
      if ( v9 == v60 )
      {
        v31 = 0;
LABEL_35:
        if ( v27 > 0xFFFFFFFE )
        {
          v33 = v27 / 0xFFFFFFFF + 1;
          v32 = v27 / v33;
        }
        else
        {
          LODWORD(v32) = v27;
          v33 = 1;
        }
        goto LABEL_37;
      }
    }
    v31 = 0;
    do
      v31 += *v26++;
    while ( v9 != v26 );
    goto LABEL_35;
  }
LABEL_18:
  sub_2240A30((unsigned __int64 *)&v84);
  v16 = v81;
  if ( v81 != (unsigned int *)v83 )
LABEL_12:
    _libc_free((unsigned __int64)v16);
}

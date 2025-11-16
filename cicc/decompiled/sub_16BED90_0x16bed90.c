// Function: sub_16BED90
// Address: 0x16bed90
//
__int64 __fastcall sub_16BED90(_BYTE *a1, __int64 a2, unsigned __int8 a3, int a4)
{
  __int64 v5; // rsi
  __m128i *v6; // rdx
  __int64 v7; // rax
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  unsigned int v10; // eax
  __int64 v11; // rdi
  unsigned int v12; // r12d
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rax
  __m128i *v16; // rdx
  __m128i v17; // xmm0
  unsigned int v18; // eax
  __int64 v20; // rsi
  __m128i *v21; // rax
  __m128i *v22; // rsi
  char v23; // al
  const char *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rax
  __m128i *v29; // rdx
  __m128i v30; // xmm0
  __m128i *v31; // rdi
  size_t v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // rax
  __m128i *v36; // rdx
  __m128i v37; // xmm0
  __int64 v38; // rax
  __m128i *v39; // rdx
  __int64 v40; // rdi
  __m128i v41; // xmm0
  __int64 v42; // rax
  __int64 v43; // rdi
  _BYTE *v44; // rax
  const char *v45; // rdi
  char *v46; // rbx
  size_t v47; // rax
  __int64 v48; // rcx
  __m128i *v49; // rsi
  const __m128i *v50; // rax
  __m128i *v51; // rsi
  const __m128i *v52; // rax
  __m128i *v53; // rsi
  const __m128i *v54; // rax
  __m128i *v55; // rsi
  const __m128i *v56; // rax
  __m128i *v57; // rsi
  const __m128i *v58; // rax
  __m128i *v59; // rsi
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rdi
  __int64 v64; // rax
  void *v65; // rdx
  __m128i *v66; // rax
  const __m128i *v67; // r8
  __m128i *v68; // rax
  unsigned int v69; // edx
  __m128i *v70; // rsi
  int v71; // [rsp+10h] [rbp-170h]
  __m128i v73; // [rsp+40h] [rbp-140h] BYREF
  __m128i *v74; // [rsp+50h] [rbp-130h] BYREF
  __m128i *v75; // [rsp+58h] [rbp-128h]
  const __m128i *v76; // [rsp+60h] [rbp-120h]
  __m128i v77; // [rsp+70h] [rbp-110h] BYREF
  _QWORD v78[2]; // [rsp+80h] [rbp-100h] BYREF
  _QWORD *v79; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v80; // [rsp+98h] [rbp-E8h]
  _QWORD v81[2]; // [rsp+A0h] [rbp-E0h] BYREF
  __m128i v82; // [rsp+B0h] [rbp-D0h] BYREF
  _QWORD v83[2]; // [rsp+C0h] [rbp-C0h] BYREF
  const char *v84; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v85; // [rsp+D8h] [rbp-A8h]
  _QWORD v86[2]; // [rsp+E0h] [rbp-A0h] BYREF
  __m128i v87; // [rsp+F0h] [rbp-90h] BYREF
  _QWORD v88[2]; // [rsp+100h] [rbp-80h] BYREF
  __m128i v89; // [rsp+110h] [rbp-70h] BYREF
  _QWORD v90[2]; // [rsp+120h] [rbp-60h] BYREF
  __m128i v91; // [rsp+130h] [rbp-50h] BYREF
  _QWORD v92[8]; // [rsp+140h] [rbp-40h] BYREF

  if ( a1 )
  {
    v77.m128i_i64[0] = (__int64)v78;
    sub_16BE2B0(v77.m128i_i64, a1, (__int64)&a1[a2]);
  }
  else
  {
    LOBYTE(v78[0]) = 0;
    v77 = (__m128i)(unsigned __int64)v78;
  }
  v80 = 0;
  v79 = v81;
  LOBYTE(v81[0]) = 0;
  v82.m128i_i64[0] = (__int64)v83;
  v82.m128i_i64[1] = 0;
  LOBYTE(v83[0]) = 0;
  v84 = (const char *)v86;
  v85 = 0;
  LOBYTE(v86[0]) = 0;
  if ( (unsigned __int8)sub_16BE410((__int64)&v84, (__int64)"xdg-open", 8, (__int64)&v82) )
  {
    v89 = v82;
    v91 = 0u;
    v92[0] = 0;
    sub_12DD210((const __m128i **)&v91, 0, &v89);
    v5 = v91.m128i_i64[1];
    v6 = &v89;
    v89 = v77;
    if ( v91.m128i_i64[1] == v92[0] )
    {
      sub_12DD210((const __m128i **)&v91, (const __m128i *)v91.m128i_i64[1], &v89);
    }
    else
    {
      if ( v91.m128i_i64[1] )
      {
        *(__m128i *)v91.m128i_i64[1] = _mm_loadu_si128(&v89);
        v5 = v91.m128i_i64[1];
      }
      v5 += 16;
      v91.m128i_i64[1] = v5;
    }
    v7 = sub_16E8CB0(&v91, v5, v6);
    v8 = *(__m128i **)(v7 + 24);
    if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 0x1Cu )
    {
      sub_16E7EE0(v7, "Trying 'xdg-open' program... ", 29);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F66B30);
      qmemcpy(&v8[1], "' program... ", 13);
      *v8 = si128;
      *(_QWORD *)(v7 + 24) += 29LL;
    }
    v10 = sub_16BE700(
            v82.m128i_i64[0],
            v82.m128i_i64[1],
            v91.m128i_i64[0],
            v91.m128i_i64[1],
            (void *)v77.m128i_i64[0],
            v77.m128i_u64[1],
            a3,
            (__int64)&v79);
    v11 = v91.m128i_i64[0];
    v12 = v10;
    if ( !(_BYTE)v10 )
      goto LABEL_21;
    if ( v91.m128i_i64[0] )
      j_j___libc_free_0(v91.m128i_i64[0], v92[0] - v91.m128i_i64[0]);
  }
  if ( (unsigned __int8)sub_16BE410((__int64)&v84, (__int64)"Graphviz", 8, (__int64)&v82) )
  {
    v91 = 0u;
    v89 = v82;
    v92[0] = 0;
    sub_12DD210((const __m128i **)&v91, 0, &v89);
    v14 = v91.m128i_i64[1];
    v89 = v77;
    if ( v91.m128i_i64[1] == v92[0] )
    {
      sub_12DD210((const __m128i **)&v91, (const __m128i *)v91.m128i_i64[1], &v89);
    }
    else
    {
      if ( v91.m128i_i64[1] )
      {
        *(__m128i *)v91.m128i_i64[1] = _mm_loadu_si128(&v89);
        v14 = v91.m128i_i64[1];
      }
      v14 += 16;
      v91.m128i_i64[1] = v14;
    }
    v15 = sub_16E8CB0(&v91, v14, v13);
    v16 = *(__m128i **)(v15 + 24);
    if ( *(_QWORD *)(v15 + 16) - (_QWORD)v16 <= 0x1Du )
    {
      sub_16E7EE0(v15, "Running 'Graphviz' program... ", 30);
    }
    else
    {
      v17 = _mm_load_si128((const __m128i *)&xmmword_3F66B40);
      qmemcpy(&v16[1], "z' program... ", 14);
      *v16 = v17;
      *(_QWORD *)(v15 + 24) += 30LL;
    }
LABEL_20:
    v18 = sub_16BE700(
            v82.m128i_i64[0],
            v82.m128i_i64[1],
            v91.m128i_i64[0],
            v91.m128i_i64[1],
            (void *)v77.m128i_i64[0],
            v77.m128i_u64[1],
            a3,
            (__int64)&v79);
    v11 = v91.m128i_i64[0];
    v12 = v18;
LABEL_21:
    if ( v11 )
      j_j___libc_free_0(v11, v92[0] - v11);
    goto LABEL_23;
  }
  if ( (unsigned __int8)sub_16BE410((__int64)&v84, (__int64)"xdot|xdot.py", 12, (__int64)&v82) )
  {
    v91 = 0u;
    v89 = v82;
    v92[0] = 0;
    sub_12DD210((const __m128i **)&v91, 0, &v89);
    v20 = v91.m128i_i64[1];
    v89 = v77;
    v21 = (__m128i *)v92[0];
    if ( v91.m128i_i64[1] == v92[0] )
    {
      sub_12DD210((const __m128i **)&v91, (const __m128i *)v91.m128i_i64[1], &v89);
      v22 = (__m128i *)v91.m128i_i64[1];
      v89.m128i_i64[1] = 2;
      v89.m128i_i64[0] = (__int64)"-f";
      if ( v92[0] != v91.m128i_i64[1] )
      {
        if ( !v91.m128i_i64[1] )
          goto LABEL_38;
        goto LABEL_37;
      }
    }
    else
    {
      if ( v91.m128i_i64[1] )
      {
        *(__m128i *)v91.m128i_i64[1] = _mm_loadu_si128(&v89);
        v20 = v91.m128i_i64[1];
        v21 = (__m128i *)v92[0];
      }
      v22 = (__m128i *)(v20 + 16);
      v89.m128i_i64[1] = 2;
      v91.m128i_i64[1] = (__int64)v22;
      v89.m128i_i64[0] = (__int64)"-f";
      if ( v22 != v21 )
      {
LABEL_37:
        *v22 = _mm_loadu_si128(&v89);
        v22 = (__m128i *)v91.m128i_i64[1];
LABEL_38:
        v91.m128i_i64[1] = (__int64)v22[1].m128i_i64;
LABEL_39:
        switch ( a4 )
        {
          case 0:
            v31 = (__m128i *)"dot";
            goto LABEL_58;
          case 1:
            v31 = (__m128i *)"fdp";
            goto LABEL_58;
          case 2:
            v31 = (__m128i *)"neato";
            goto LABEL_58;
          case 3:
            v31 = (__m128i *)"twopi";
            goto LABEL_58;
          case 4:
            v31 = (__m128i *)"circo";
LABEL_58:
            v89.m128i_i64[0] = (__int64)v31;
            v32 = strlen(v31->m128i_i8);
            v34 = v91.m128i_i64[1];
            v89.m128i_i64[1] = v32;
            if ( v91.m128i_i64[1] == v92[0] )
            {
              v31 = &v91;
              sub_12DD210((const __m128i **)&v91, (const __m128i *)v91.m128i_i64[1], &v89);
            }
            else
            {
              if ( v91.m128i_i64[1] )
              {
                *(__m128i *)v91.m128i_i64[1] = _mm_loadu_si128(&v89);
                v34 = v91.m128i_i64[1];
              }
              v34 += 16;
              v91.m128i_i64[1] = v34;
            }
            v35 = sub_16E8CB0(v31, v34, v33);
            v36 = *(__m128i **)(v35 + 24);
            if ( *(_QWORD *)(v35 + 16) - (_QWORD)v36 <= 0x1Cu )
            {
              sub_16E7EE0(v35, "Running 'xdot.py' program... ", 29);
            }
            else
            {
              v37 = _mm_load_si128((const __m128i *)&xmmword_3F66B50);
              qmemcpy(&v36[1], "' program... ", 13);
              *v36 = v37;
              *(_QWORD *)(v35 + 24) += 29LL;
            }
            goto LABEL_20;
          default:
            goto LABEL_165;
        }
      }
    }
    sub_12DD210((const __m128i **)&v91, v22, &v89);
    goto LABEL_39;
  }
  if ( (unsigned __int8)sub_16BE410((__int64)&v84, (__int64)"gv", 2, (__int64)&v82) )
  {
    LOBYTE(v88[0]) = 0;
    v87.m128i_i64[0] = (__int64)v88;
    v87.m128i_i64[1] = 0;
    v71 = 3;
  }
  else
  {
    v23 = sub_16BE410((__int64)&v84, (__int64)"xdg-open", 8, (__int64)&v82);
    LOBYTE(v88[0]) = 0;
    v87.m128i_i64[1] = 0;
    v87.m128i_i64[0] = (__int64)v88;
    if ( !v23 )
    {
LABEL_46:
      v24 = "dotty";
      if ( (unsigned __int8)sub_16BE410((__int64)&v84, (__int64)"dotty", 5, (__int64)&v82) )
      {
        v91 = 0u;
        v89 = v82;
        v92[0] = 0;
        sub_12DD210((const __m128i **)&v91, 0, &v89);
        v27 = v91.m128i_i64[1];
        v89 = v77;
        if ( v91.m128i_i64[1] == v92[0] )
        {
          sub_12DD210((const __m128i **)&v91, (const __m128i *)v91.m128i_i64[1], &v89);
        }
        else
        {
          if ( v91.m128i_i64[1] )
          {
            *(__m128i *)v91.m128i_i64[1] = _mm_loadu_si128(&v89);
            v27 = v91.m128i_i64[1];
          }
          v27 += 16;
          v91.m128i_i64[1] = v27;
        }
        v28 = sub_16E8CB0(&v91, v27, v26);
        v29 = *(__m128i **)(v28 + 24);
        if ( *(_QWORD *)(v28 + 16) - (_QWORD)v29 <= 0x1Au )
        {
          sub_16E7EE0(v28, "Running 'dotty' program... ", 27);
        }
        else
        {
          v30 = _mm_load_si128((const __m128i *)&xmmword_3F66B60);
          qmemcpy(&v29[1], "program... ", 11);
          *v29 = v30;
          *(_QWORD *)(v28 + 24) += 27LL;
        }
        v12 = sub_16BE700(
                v82.m128i_i64[0],
                v82.m128i_i64[1],
                v91.m128i_i64[0],
                v91.m128i_i64[1],
                (void *)v77.m128i_i64[0],
                v77.m128i_u64[1],
                a3,
                (__int64)&v79);
        if ( v91.m128i_i64[0] )
          j_j___libc_free_0(v91.m128i_i64[0], v92[0] - v91.m128i_i64[0]);
      }
      else
      {
        v38 = sub_16E8CB0(&v84, "dotty", v25);
        v39 = *(__m128i **)(v38 + 24);
        v40 = v38;
        if ( *(_QWORD *)(v38 + 16) - (_QWORD)v39 <= 0x33u )
        {
          v24 = "Error: Couldn't find a usable graph viewer program:\n";
          sub_16E7EE0(v38, "Error: Couldn't find a usable graph viewer program:\n", 52);
        }
        else
        {
          v41 = _mm_load_si128((const __m128i *)&xmmword_3F66B70);
          v39[3].m128i_i32[0] = 171601249;
          *v39 = v41;
          v39[1] = _mm_load_si128((const __m128i *)&xmmword_3F66B80);
          v39[2] = _mm_load_si128((const __m128i *)&xmmword_3F66B90);
          *(_QWORD *)(v38 + 24) += 52LL;
        }
        v42 = sub_16E8CB0(v40, v24, v39);
        v43 = sub_16E7EE0(v42, v84, v85);
        v44 = *(_BYTE **)(v43 + 24);
        if ( *(_BYTE **)(v43 + 16) == v44 )
        {
          sub_16E7EE0(v43, "\n", 1);
        }
        else
        {
          *v44 = 10;
          ++*(_QWORD *)(v43 + 24);
        }
        v12 = 1;
      }
      goto LABEL_55;
    }
    v71 = 2;
  }
  switch ( a4 )
  {
    case 0:
      v46 = "dot";
      v45 = "dot";
      goto LABEL_83;
    case 1:
      v45 = "fdp";
      goto LABEL_82;
    case 2:
      v45 = "neato";
      goto LABEL_82;
    case 3:
      v45 = "twopi";
      goto LABEL_82;
    case 4:
      v45 = "circo";
LABEL_82:
      v46 = (char *)v45;
LABEL_83:
      v47 = strlen(v45);
      if ( !(unsigned __int8)sub_16BE410((__int64)&v84, (__int64)v46, v47, (__int64)&v87)
        && !(unsigned __int8)sub_16BE410((__int64)&v84, (__int64)"dot|fdp|neato|twopi|circo", 25, (__int64)&v87) )
      {
        goto LABEL_46;
      }
      v89.m128i_i64[0] = (__int64)v90;
      sub_16BE360(v89.m128i_i64, v77.m128i_i64[0], v77.m128i_i64[0] + v77.m128i_i64[1]);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v89.m128i_i64[1]) <= 2 )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(&v89, ".ps", 3, v48);
      v74 = 0;
      v75 = 0;
      v91 = v87;
      v76 = 0;
      sub_12DD210((const __m128i **)&v74, 0, &v91);
      v49 = v75;
      v91.m128i_i64[1] = 4;
      v91.m128i_i64[0] = (__int64)"-Tps";
      v50 = v76;
      if ( v75 == v76 )
      {
        sub_12DD210((const __m128i **)&v74, v75, &v91);
        v51 = v75;
        v52 = v76;
        v91.m128i_i64[0] = (__int64)"-Nfontname=Courier";
        v91.m128i_i64[1] = 18;
        if ( v75 != v76 )
        {
          if ( !v75 )
          {
LABEL_91:
            v53 = v51 + 1;
            v91.m128i_i64[1] = 13;
            v75 = v53;
            v91.m128i_i64[0] = (__int64)"-Gsize=7.5,10";
            if ( v53 != v52 )
              goto LABEL_92;
            goto LABEL_144;
          }
LABEL_90:
          *v51 = _mm_loadu_si128(&v91);
          v51 = v75;
          v52 = v76;
          goto LABEL_91;
        }
      }
      else
      {
        if ( v75 )
        {
          *v75 = _mm_loadu_si128(&v91);
          v49 = v75;
          v50 = v76;
        }
        v51 = v49 + 1;
        v91.m128i_i64[1] = 18;
        v75 = v51;
        v91.m128i_i64[0] = (__int64)"-Nfontname=Courier";
        if ( v51 != v50 )
          goto LABEL_90;
      }
      sub_12DD210((const __m128i **)&v74, v51, &v91);
      v53 = v75;
      v54 = v76;
      v91.m128i_i64[0] = (__int64)"-Gsize=7.5,10";
      v91.m128i_i64[1] = 13;
      if ( v75 != v76 )
      {
        if ( !v75 )
        {
LABEL_93:
          v55 = v53 + 1;
          v75 = v55;
          v91 = v77;
          if ( v54 != v55 )
            goto LABEL_94;
          goto LABEL_141;
        }
LABEL_92:
        *v53 = _mm_loadu_si128(&v91);
        v53 = v75;
        v54 = v76;
        goto LABEL_93;
      }
LABEL_144:
      sub_12DD210((const __m128i **)&v74, v53, &v91);
      v55 = v75;
      v56 = v76;
      v91 = v77;
      if ( v76 != v75 )
      {
        if ( !v75 )
        {
LABEL_95:
          v57 = v55 + 1;
          v91.m128i_i64[1] = 2;
          v75 = v57;
          v91.m128i_i64[0] = (__int64)"-o";
          if ( v56 != v57 )
            goto LABEL_96;
          goto LABEL_153;
        }
LABEL_94:
        *v55 = _mm_loadu_si128(&v91);
        v55 = v75;
        v56 = v76;
        goto LABEL_95;
      }
LABEL_141:
      sub_12DD210((const __m128i **)&v74, v55, &v91);
      v57 = v75;
      v58 = v76;
      v91.m128i_i64[0] = (__int64)"-o";
      v91.m128i_i64[1] = 2;
      if ( v75 != v76 )
      {
        if ( !v75 )
        {
LABEL_97:
          v59 = v57 + 1;
          v75 = v59;
          v60 = v89.m128i_i64[1];
          v91 = v89;
          if ( v59 != v58 )
          {
LABEL_98:
            *v59 = _mm_loadu_si128(&v91);
            v59 = v75;
LABEL_99:
            v75 = ++v59;
            goto LABEL_100;
          }
          goto LABEL_140;
        }
LABEL_96:
        *v57 = _mm_loadu_si128(&v91);
        v57 = v75;
        v58 = v76;
        goto LABEL_97;
      }
LABEL_153:
      sub_12DD210((const __m128i **)&v74, v57, &v91);
      v59 = v75;
      v91 = v89;
      if ( v75 != v76 )
      {
        if ( !v75 )
          goto LABEL_99;
        goto LABEL_98;
      }
LABEL_140:
      sub_12DD210((const __m128i **)&v74, v59, &v91);
LABEL_100:
      v61 = sub_16E8CB0(&v74, v59, v60);
      v62 = *(_QWORD *)(v61 + 24);
      v63 = v61;
      if ( (unsigned __int64)(*(_QWORD *)(v61 + 16) - v62) <= 8 )
      {
        v63 = sub_16E7EE0(v61, "Running '", 9);
      }
      else
      {
        *(_BYTE *)(v62 + 8) = 39;
        *(_QWORD *)v62 = 0x20676E696E6E7552LL;
        *(_QWORD *)(v61 + 24) += 9LL;
      }
      v64 = sub_16E7EE0(v63, (const char *)v87.m128i_i64[0], v87.m128i_i64[1]);
      v65 = *(void **)(v64 + 24);
      if ( *(_QWORD *)(v64 + 16) - (_QWORD)v65 <= 0xCu )
      {
        sub_16E7EE0(v64, "' program... ", 13);
      }
      else
      {
        qmemcpy(v65, "' program... ", 13);
        *(_QWORD *)(v64 + 24) += 13LL;
      }
      v12 = sub_16BE700(
              v87.m128i_i64[0],
              v87.m128i_i64[1],
              (__int64)v74,
              (__int64)v75,
              (void *)v77.m128i_i64[0],
              v77.m128i_u64[1],
              1u,
              (__int64)&v79);
      if ( (_BYTE)v12 )
        goto LABEL_119;
      v66 = v74;
      v91 = (__m128i)(unsigned __int64)v92;
      LOBYTE(v92[0]) = 0;
      if ( v74 != v75 )
        v75 = v74;
      v67 = v76;
      v73 = v82;
      if ( v74 == v76 )
      {
        sub_12DD210((const __m128i **)&v74, v76, &v73);
        v68 = v75;
        v67 = v76;
      }
      else
      {
        if ( v74 )
        {
          *v74 = _mm_loadu_si128(&v73);
          v66 = v75;
          v67 = v76;
        }
        v68 = v66 + 1;
        v75 = v68;
      }
      if ( v71 == 3 )
      {
        v73.m128i_i64[1] = 9;
        v73.m128i_i64[0] = (__int64)"--spartan";
        if ( v67 == v68 )
        {
          sub_12DD210((const __m128i **)&v74, v67, &v73);
          v70 = v75;
          v73 = v89;
          if ( v75 != v76 )
          {
            if ( !v75 )
              goto LABEL_138;
            goto LABEL_137;
          }
        }
        else
        {
          if ( v68 )
          {
            *v68 = _mm_loadu_si128(&v73);
            v68 = v75;
            v67 = v76;
          }
          v70 = v68 + 1;
          v75 = v68 + 1;
          v73 = v89;
          if ( v67 != &v68[1] )
          {
LABEL_137:
            *v70 = _mm_loadu_si128(&v73);
            v70 = v75;
LABEL_138:
            v75 = v70 + 1;
LABEL_139:
            v69 = a3;
            goto LABEL_117;
          }
        }
        sub_12DD210((const __m128i **)&v74, v70, &v73);
        goto LABEL_139;
      }
      v73 = v89;
      if ( v67 == v68 )
      {
        sub_12DD210((const __m128i **)&v74, v67, &v73);
      }
      else
      {
        if ( v68 )
        {
          *v68 = _mm_loadu_si128(&v73);
          v68 = v75;
        }
        v75 = v68 + 1;
      }
      v69 = 0;
LABEL_117:
      v80 = 0;
      *(_BYTE *)v79 = 0;
      v12 = sub_16BE700(
              v82.m128i_i64[0],
              v82.m128i_i64[1],
              (__int64)v74,
              (__int64)v75,
              (void *)v89.m128i_i64[0],
              v89.m128i_u64[1],
              v69,
              (__int64)&v79);
      if ( (_QWORD *)v91.m128i_i64[0] != v92 )
        j_j___libc_free_0(v91.m128i_i64[0], v92[0] + 1LL);
LABEL_119:
      if ( v74 )
        j_j___libc_free_0(v74, (char *)v76 - (char *)v74);
      if ( (_QWORD *)v89.m128i_i64[0] != v90 )
        j_j___libc_free_0(v89.m128i_i64[0], v90[0] + 1LL);
LABEL_55:
      if ( (_QWORD *)v87.m128i_i64[0] != v88 )
        j_j___libc_free_0(v87.m128i_i64[0], v88[0] + 1LL);
LABEL_23:
      if ( v84 != (const char *)v86 )
        j_j___libc_free_0(v84, v86[0] + 1LL);
      if ( (_QWORD *)v82.m128i_i64[0] != v83 )
        j_j___libc_free_0(v82.m128i_i64[0], v83[0] + 1LL);
      if ( v79 != v81 )
        j_j___libc_free_0(v79, v81[0] + 1LL);
      if ( (_QWORD *)v77.m128i_i64[0] != v78 )
        j_j___libc_free_0(v77.m128i_i64[0], v78[0] + 1LL);
      return v12;
    default:
LABEL_165:
      JUMPOUT(0x41A81C);
  }
}

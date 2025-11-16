// Function: sub_C67930
// Address: 0xc67930
//
__int64 __fastcall sub_C67930(_BYTE *a1, __int64 a2, unsigned __int8 a3, int a4)
{
  __int64 v4; // rsi
  __int64 v5; // rax
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  unsigned int v8; // eax
  __int64 v9; // rdi
  unsigned int v10; // r12d
  __int64 v11; // rsi
  __int64 v12; // rax
  __m128i *v13; // rdx
  __m128i v14; // xmm0
  unsigned int v15; // eax
  __int64 v17; // r14
  const __m128i *v18; // rsi
  __m128i *v19; // r14
  __m128i *v20; // r14
  char v21; // al
  const char *v22; // rsi
  __int64 v23; // rsi
  __int64 v24; // rax
  __m128i *v25; // rdx
  __m128i v26; // xmm0
  __m128i *v27; // rdi
  __int64 v28; // rax
  __m128i *v29; // rdx
  __m128i v30; // xmm0
  __int64 v31; // rax
  __m128i *v32; // rdx
  __int64 v33; // rdi
  __m128i v34; // xmm0
  __int64 v35; // rax
  __int64 v36; // rdi
  _BYTE *v37; // rax
  const char *v38; // rdi
  char *v39; // rsi
  size_t v40; // rax
  __int64 v41; // rcx
  __m128i *v42; // rsi
  const __m128i *v43; // r8
  __m128i *v44; // rsi
  __m128i *v45; // rsi
  const __m128i *v46; // r8
  __m128i *v47; // rsi
  __m128i *v48; // rsi
  __m128i *v49; // rsi
  __m128i *v50; // rsi
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rdi
  __int64 v54; // rax
  void *v55; // rdx
  __m128i *v56; // rsi
  const __m128i *v57; // rdx
  __m128i *v58; // rsi
  unsigned int v59; // edx
  __m128i *v60; // rsi
  int v61; // [rsp+10h] [rbp-170h]
  __m128i v64; // [rsp+40h] [rbp-140h] BYREF
  __m128i *v65; // [rsp+50h] [rbp-130h] BYREF
  __m128i *v66; // [rsp+58h] [rbp-128h]
  const __m128i *v67; // [rsp+60h] [rbp-120h]
  __m128i v68; // [rsp+70h] [rbp-110h] BYREF
  _QWORD v69[2]; // [rsp+80h] [rbp-100h] BYREF
  _QWORD *v70; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v71; // [rsp+98h] [rbp-E8h]
  _QWORD v72[2]; // [rsp+A0h] [rbp-E0h] BYREF
  __m128i v73; // [rsp+B0h] [rbp-D0h] BYREF
  _QWORD v74[2]; // [rsp+C0h] [rbp-C0h] BYREF
  _QWORD *v75; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v76; // [rsp+D8h] [rbp-A8h]
  _QWORD v77[2]; // [rsp+E0h] [rbp-A0h] BYREF
  __m128i v78; // [rsp+F0h] [rbp-90h] BYREF
  _QWORD v79[2]; // [rsp+100h] [rbp-80h] BYREF
  __m128i v80; // [rsp+110h] [rbp-70h] BYREF
  _QWORD v81[2]; // [rsp+120h] [rbp-60h] BYREF
  __m128i v82; // [rsp+130h] [rbp-50h] BYREF
  const __m128i *v83[8]; // [rsp+140h] [rbp-40h] BYREF

  v68.m128i_i64[0] = (__int64)v69;
  sub_C66AC0(v68.m128i_i64, a1, (__int64)&a1[a2]);
  v71 = 0;
  v70 = v72;
  LOBYTE(v72[0]) = 0;
  v73.m128i_i64[0] = (__int64)v74;
  v73.m128i_i64[1] = 0;
  LOBYTE(v74[0]) = 0;
  v75 = v77;
  v76 = 0;
  LOBYTE(v77[0]) = 0;
  if ( (unsigned __int8)sub_C66C20((__int64)&v75, (__int64)"xdg-open", 8, (__int64)&v73) )
  {
    v80 = v73;
    v82 = 0u;
    v83[0] = 0;
    sub_C677B0((const __m128i **)&v82, 0, &v80);
    v4 = v82.m128i_i64[1];
    v80 = v68;
    if ( v83[0] == (const __m128i *)v82.m128i_i64[1] )
    {
      sub_C677B0((const __m128i **)&v82, (const __m128i *)v82.m128i_i64[1], &v80);
    }
    else
    {
      if ( v82.m128i_i64[1] )
      {
        *(__m128i *)v82.m128i_i64[1] = _mm_loadu_si128(&v80);
        v4 = v82.m128i_i64[1];
      }
      v4 += 16;
      v82.m128i_i64[1] = v4;
    }
    v5 = sub_CB72A0(&v82, v4);
    v6 = *(__m128i **)(v5 + 32);
    if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0x1Cu )
    {
      sub_CB6200(v5, "Trying 'xdg-open' program... ", 29);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F66B30);
      qmemcpy(&v6[1], "' program... ", 13);
      *v6 = si128;
      *(_QWORD *)(v5 + 32) += 29LL;
    }
    v8 = sub_C66F30(
           v73.m128i_i64[0],
           v73.m128i_i64[1],
           v82.m128i_i64[0],
           v82.m128i_i64[1],
           (const void *)v68.m128i_i64[0],
           v68.m128i_u64[1],
           a3,
           &v70);
    v9 = v82.m128i_i64[0];
    v10 = v8;
    if ( !(_BYTE)v8 )
      goto LABEL_19;
    if ( v82.m128i_i64[0] )
      j_j___libc_free_0(v82.m128i_i64[0], (char *)v83[0] - v82.m128i_i64[0]);
  }
  if ( (unsigned __int8)sub_C66C20((__int64)&v75, (__int64)"Graphviz", 8, (__int64)&v73) )
  {
    v82 = 0u;
    v80 = v73;
    v83[0] = 0;
    sub_C677B0((const __m128i **)&v82, 0, &v80);
    v11 = v82.m128i_i64[1];
    v80 = v68;
    if ( v83[0] == (const __m128i *)v82.m128i_i64[1] )
    {
      sub_C677B0((const __m128i **)&v82, (const __m128i *)v82.m128i_i64[1], &v80);
    }
    else
    {
      if ( v82.m128i_i64[1] )
      {
        *(__m128i *)v82.m128i_i64[1] = _mm_loadu_si128(&v80);
        v11 = v82.m128i_i64[1];
      }
      v11 += 16;
      v82.m128i_i64[1] = v11;
    }
    v12 = sub_CB72A0(&v82, v11);
    v13 = *(__m128i **)(v12 + 32);
    if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 0x1Du )
    {
      sub_CB6200(v12, "Running 'Graphviz' program... ", 30);
    }
    else
    {
      v14 = _mm_load_si128((const __m128i *)&xmmword_3F66B40);
      qmemcpy(&v13[1], "z' program... ", 14);
      *v13 = v14;
      *(_QWORD *)(v12 + 32) += 30LL;
    }
LABEL_18:
    v15 = sub_C66F30(
            v73.m128i_i64[0],
            v73.m128i_i64[1],
            v82.m128i_i64[0],
            v82.m128i_i64[1],
            (const void *)v68.m128i_i64[0],
            v68.m128i_u64[1],
            a3,
            &v70);
    v9 = v82.m128i_i64[0];
    v10 = v15;
LABEL_19:
    if ( v9 )
      j_j___libc_free_0(v9, (char *)v83[0] - v9);
    goto LABEL_21;
  }
  if ( (unsigned __int8)sub_C66C20((__int64)&v75, (__int64)"xdot|xdot.py", 12, (__int64)&v73) )
  {
    v82 = 0u;
    v80 = v73;
    v83[0] = 0;
    sub_C677B0((const __m128i **)&v82, 0, &v80);
    v17 = v82.m128i_i64[1];
    v18 = v83[0];
    v80 = v68;
    if ( v83[0] == (const __m128i *)v82.m128i_i64[1] )
    {
      sub_C677B0((const __m128i **)&v82, v83[0], &v80);
      v19 = (__m128i *)v82.m128i_i64[1];
      v80.m128i_i64[1] = 2;
      v80.m128i_i64[0] = (__int64)"-f";
      if ( v83[0] != (const __m128i *)v82.m128i_i64[1] )
      {
        if ( !v82.m128i_i64[1] )
          goto LABEL_36;
        goto LABEL_35;
      }
      v18 = (const __m128i *)v82.m128i_i64[1];
    }
    else
    {
      if ( v82.m128i_i64[1] )
      {
        *(__m128i *)v82.m128i_i64[1] = _mm_loadu_si128(&v80);
        v17 = v82.m128i_i64[1];
        v18 = v83[0];
      }
      v19 = (__m128i *)(v17 + 16);
      v80.m128i_i64[1] = 2;
      v82.m128i_i64[1] = (__int64)v19;
      v80.m128i_i64[0] = (__int64)"-f";
      if ( v19 != v18 )
      {
LABEL_35:
        *v19 = _mm_loadu_si128(&v80);
        v19 = (__m128i *)v82.m128i_i64[1];
LABEL_36:
        v20 = v19 + 1;
        v82.m128i_i64[1] = (__int64)v20;
LABEL_37:
        switch ( a4 )
        {
          case 0:
            v27 = (__m128i *)"dot";
            goto LABEL_53;
          case 1:
            v27 = (__m128i *)"fdp";
            goto LABEL_53;
          case 2:
            v27 = (__m128i *)"neato";
            goto LABEL_53;
          case 3:
            v27 = (__m128i *)"twopi";
            goto LABEL_53;
          case 4:
            v27 = (__m128i *)"circo";
LABEL_53:
            v80.m128i_i64[0] = (__int64)v27;
            v80.m128i_i64[1] = strlen(v27->m128i_i8);
            if ( v20 == v83[0] )
            {
              v18 = v20;
              v27 = &v82;
              sub_C677B0((const __m128i **)&v82, v20, &v80);
            }
            else
            {
              if ( v20 )
              {
                *v20 = _mm_loadu_si128(&v80);
                v20 = (__m128i *)v82.m128i_i64[1];
              }
              v82.m128i_i64[1] = (__int64)v20[1].m128i_i64;
            }
            v28 = sub_CB72A0(v27, v18);
            v29 = *(__m128i **)(v28 + 32);
            if ( *(_QWORD *)(v28 + 24) - (_QWORD)v29 <= 0x1Cu )
            {
              sub_CB6200(v28, "Running 'xdot.py' program... ", 29);
            }
            else
            {
              v30 = _mm_load_si128((const __m128i *)&xmmword_3F66B50);
              qmemcpy(&v29[1], "' program... ", 13);
              *v29 = v30;
              *(_QWORD *)(v28 + 32) += 29LL;
            }
            goto LABEL_18;
          default:
            goto LABEL_164;
        }
      }
    }
    sub_C677B0((const __m128i **)&v82, v18, &v80);
    v20 = (__m128i *)v82.m128i_i64[1];
    goto LABEL_37;
  }
  if ( (unsigned __int8)sub_C66C20((__int64)&v75, (__int64)"gv", 2, (__int64)&v73) )
  {
    LOBYTE(v79[0]) = 0;
    v78.m128i_i64[0] = (__int64)v79;
    v78.m128i_i64[1] = 0;
    v61 = 3;
  }
  else
  {
    v21 = sub_C66C20((__int64)&v75, (__int64)"xdg-open", 8, (__int64)&v73);
    LOBYTE(v79[0]) = 0;
    v78.m128i_i64[1] = 0;
    v78.m128i_i64[0] = (__int64)v79;
    if ( !v21 )
    {
LABEL_43:
      v22 = "dotty";
      if ( (unsigned __int8)sub_C66C20((__int64)&v75, (__int64)"dotty", 5, (__int64)&v73) )
      {
        v82 = 0u;
        v80 = v73;
        v83[0] = 0;
        sub_C677B0((const __m128i **)&v82, 0, &v80);
        v23 = v82.m128i_i64[1];
        v80 = v68;
        if ( v83[0] == (const __m128i *)v82.m128i_i64[1] )
        {
          sub_C677B0((const __m128i **)&v82, (const __m128i *)v82.m128i_i64[1], &v80);
        }
        else
        {
          if ( v82.m128i_i64[1] )
          {
            *(__m128i *)v82.m128i_i64[1] = _mm_loadu_si128(&v80);
            v23 = v82.m128i_i64[1];
          }
          v23 += 16;
          v82.m128i_i64[1] = v23;
        }
        v24 = sub_CB72A0(&v82, v23);
        v25 = *(__m128i **)(v24 + 32);
        if ( *(_QWORD *)(v24 + 24) - (_QWORD)v25 <= 0x1Au )
        {
          sub_CB6200(v24, "Running 'dotty' program... ", 27);
        }
        else
        {
          v26 = _mm_load_si128((const __m128i *)&xmmword_3F66B60);
          qmemcpy(&v25[1], "program... ", 11);
          *v25 = v26;
          *(_QWORD *)(v24 + 32) += 27LL;
        }
        v10 = sub_C66F30(
                v73.m128i_i64[0],
                v73.m128i_i64[1],
                v82.m128i_i64[0],
                v82.m128i_i64[1],
                (const void *)v68.m128i_i64[0],
                v68.m128i_u64[1],
                a3,
                &v70);
        if ( v82.m128i_i64[0] )
          j_j___libc_free_0(v82.m128i_i64[0], (char *)v83[0] - v82.m128i_i64[0]);
      }
      else
      {
        v31 = sub_CB72A0(&v75, "dotty");
        v32 = *(__m128i **)(v31 + 32);
        v33 = v31;
        if ( *(_QWORD *)(v31 + 24) - (_QWORD)v32 <= 0x33u )
        {
          v22 = "Error: Couldn't find a usable graph viewer program:\n";
          sub_CB6200(v31, "Error: Couldn't find a usable graph viewer program:\n", 52);
        }
        else
        {
          v34 = _mm_load_si128((const __m128i *)&xmmword_3F66B70);
          v32[3].m128i_i32[0] = 171601249;
          *v32 = v34;
          v32[1] = _mm_load_si128((const __m128i *)&xmmword_3F66B80);
          v32[2] = _mm_load_si128((const __m128i *)&xmmword_3F66B90);
          *(_QWORD *)(v31 + 32) += 52LL;
        }
        v35 = sub_CB72A0(v33, v22);
        v36 = sub_CB6200(v35, v75, v76);
        v37 = *(_BYTE **)(v36 + 32);
        if ( *(_BYTE **)(v36 + 24) == v37 )
        {
          sub_CB6200(v36, "\n", 1);
        }
        else
        {
          *v37 = 10;
          ++*(_QWORD *)(v36 + 32);
        }
        v10 = 1;
      }
      goto LABEL_69;
    }
    v61 = 2;
  }
  switch ( a4 )
  {
    case 0:
      v39 = "dot";
      v38 = "dot";
      goto LABEL_82;
    case 1:
      v38 = "fdp";
      goto LABEL_81;
    case 2:
      v38 = "neato";
      goto LABEL_81;
    case 3:
      v38 = "twopi";
      goto LABEL_81;
    case 4:
      v38 = "circo";
LABEL_81:
      v39 = (char *)v38;
LABEL_82:
      v40 = strlen(v38);
      if ( !(unsigned __int8)sub_C66C20((__int64)&v75, (__int64)v39, v40, (__int64)&v78)
        && !(unsigned __int8)sub_C66C20((__int64)&v75, (__int64)"dot|fdp|neato|twopi|circo", 25, (__int64)&v78) )
      {
        goto LABEL_43;
      }
      v80.m128i_i64[0] = (__int64)v81;
      sub_C66B70(v80.m128i_i64, v68.m128i_i64[0], v68.m128i_i64[0] + v68.m128i_i64[1]);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v80.m128i_i64[1]) <= 2 )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(&v80, ".ps", 3, v41);
      v65 = 0;
      v66 = 0;
      v82 = v78;
      v67 = 0;
      sub_C677B0((const __m128i **)&v65, 0, &v82);
      v42 = v66;
      v43 = v67;
      v82.m128i_i64[0] = (__int64)"-Tps";
      v82.m128i_i64[1] = 4;
      if ( v67 == v66 )
      {
        sub_C677B0((const __m128i **)&v65, v67, &v82);
        v44 = v66;
        v43 = v67;
        v82.m128i_i64[0] = (__int64)"-Nfontname=Courier";
        v82.m128i_i64[1] = 18;
        if ( v67 != v66 )
        {
          if ( !v66 )
          {
LABEL_90:
            v45 = v44 + 1;
            v82.m128i_i64[1] = 13;
            v66 = v45;
            v82.m128i_i64[0] = (__int64)"-Gsize=7.5,10";
            if ( v43 != v45 )
              goto LABEL_91;
            goto LABEL_150;
          }
LABEL_89:
          *v44 = _mm_loadu_si128(&v82);
          v44 = v66;
          v43 = v67;
          goto LABEL_90;
        }
      }
      else
      {
        if ( v66 )
        {
          *v66 = _mm_loadu_si128(&v82);
          v43 = v67;
          v42 = v66;
        }
        v44 = v42 + 1;
        v82.m128i_i64[1] = 18;
        v66 = v44;
        v82.m128i_i64[0] = (__int64)"-Nfontname=Courier";
        if ( v44 != v43 )
          goto LABEL_89;
      }
      sub_C677B0((const __m128i **)&v65, v43, &v82);
      v45 = v66;
      v46 = v67;
      v82.m128i_i64[0] = (__int64)"-Gsize=7.5,10";
      v82.m128i_i64[1] = 13;
      if ( v67 != v66 )
      {
        if ( !v66 )
        {
LABEL_92:
          v47 = v45 + 1;
          v66 = v47;
          v82 = v68;
          if ( v47 != v46 )
            goto LABEL_93;
          goto LABEL_147;
        }
LABEL_91:
        *v45 = _mm_loadu_si128(&v82);
        v45 = v66;
        v46 = v67;
        goto LABEL_92;
      }
LABEL_150:
      sub_C677B0((const __m128i **)&v65, v45, &v82);
      v47 = v66;
      v46 = v67;
      v82 = v68;
      if ( v67 != v66 )
      {
        if ( !v66 )
        {
LABEL_94:
          v48 = v47 + 1;
          v82.m128i_i64[1] = 2;
          v66 = v48;
          v82.m128i_i64[0] = (__int64)"-o";
          if ( v48 != v46 )
            goto LABEL_95;
          goto LABEL_144;
        }
LABEL_93:
        *v47 = _mm_loadu_si128(&v82);
        v47 = v66;
        v46 = v67;
        goto LABEL_94;
      }
LABEL_147:
      sub_C677B0((const __m128i **)&v65, v46, &v82);
      v48 = v66;
      v46 = v67;
      v82.m128i_i64[0] = (__int64)"-o";
      v82.m128i_i64[1] = 2;
      if ( v67 != v66 )
      {
        if ( !v66 )
        {
LABEL_96:
          v49 = v48 + 1;
          v66 = v49;
          v82 = v80;
          if ( v49 != v46 )
          {
LABEL_97:
            *v49 = _mm_loadu_si128(&v82);
            v49 = v66;
LABEL_98:
            v50 = v49 + 1;
            v66 = v50;
            goto LABEL_99;
          }
          goto LABEL_140;
        }
LABEL_95:
        *v48 = _mm_loadu_si128(&v82);
        v46 = v67;
        v48 = v66;
        goto LABEL_96;
      }
LABEL_144:
      sub_C677B0((const __m128i **)&v65, v46, &v82);
      v49 = v66;
      v82 = v80;
      if ( v67 != v66 )
      {
        if ( !v66 )
          goto LABEL_98;
        goto LABEL_97;
      }
      v46 = v66;
LABEL_140:
      v50 = (__m128i *)v46;
      sub_C677B0((const __m128i **)&v65, v46, &v82);
LABEL_99:
      v51 = sub_CB72A0(&v65, v50);
      v52 = *(_QWORD *)(v51 + 32);
      v53 = v51;
      if ( (unsigned __int64)(*(_QWORD *)(v51 + 24) - v52) <= 8 )
      {
        v53 = sub_CB6200(v51, "Running '", 9);
      }
      else
      {
        *(_BYTE *)(v52 + 8) = 39;
        *(_QWORD *)v52 = 0x20676E696E6E7552LL;
        *(_QWORD *)(v51 + 32) += 9LL;
      }
      v54 = sub_CB6200(v53, v78.m128i_i64[0], v78.m128i_i64[1]);
      v55 = *(void **)(v54 + 32);
      if ( *(_QWORD *)(v54 + 24) - (_QWORD)v55 <= 0xCu )
      {
        sub_CB6200(v54, "' program... ", 13);
      }
      else
      {
        qmemcpy(v55, "' program... ", 13);
        *(_QWORD *)(v54 + 32) += 13LL;
      }
      v10 = sub_C66F30(
              v78.m128i_i64[0],
              v78.m128i_i64[1],
              (__int64)v65,
              (__int64)v66,
              (const void *)v68.m128i_i64[0],
              v68.m128i_u64[1],
              1u,
              &v70);
      if ( (_BYTE)v10 )
        goto LABEL_118;
      v56 = v65;
      v82 = (__m128i)(unsigned __int64)v83;
      LOBYTE(v83[0]) = 0;
      if ( v65 != v66 )
        v66 = v65;
      v57 = v67;
      v64 = v73;
      if ( v67 == v65 )
      {
        sub_C677B0((const __m128i **)&v65, v65, &v64);
        v58 = v66;
        v57 = v67;
      }
      else
      {
        if ( v65 )
        {
          *v65 = _mm_loadu_si128(&v64);
          v56 = v66;
          v57 = v67;
        }
        v58 = v56 + 1;
        v66 = v58;
      }
      if ( v61 == 3 )
      {
        v64.m128i_i64[1] = 9;
        v64.m128i_i64[0] = (__int64)"--spartan";
        if ( v58 == v57 )
        {
          sub_C677B0((const __m128i **)&v65, v58, &v64);
          v60 = v66;
          v64 = v80;
          if ( v67 != v66 )
          {
            if ( !v66 )
              goto LABEL_137;
            goto LABEL_136;
          }
        }
        else
        {
          if ( v58 )
          {
            *v58 = _mm_loadu_si128(&v64);
            v57 = v67;
            v58 = v66;
          }
          v60 = v58 + 1;
          v66 = v60;
          v64 = v80;
          if ( v57 != v60 )
          {
LABEL_136:
            *v60 = _mm_loadu_si128(&v64);
            v60 = v66;
LABEL_137:
            v66 = v60 + 1;
LABEL_138:
            v59 = a3;
            goto LABEL_116;
          }
        }
        sub_C677B0((const __m128i **)&v65, v60, &v64);
        goto LABEL_138;
      }
      v64 = v80;
      if ( v58 == v57 )
      {
        sub_C677B0((const __m128i **)&v65, v58, &v64);
      }
      else
      {
        if ( v58 )
        {
          *v58 = _mm_loadu_si128(&v64);
          v58 = v66;
        }
        v66 = v58 + 1;
      }
      v59 = 0;
LABEL_116:
      v71 = 0;
      *(_BYTE *)v70 = 0;
      v10 = sub_C66F30(
              v73.m128i_i64[0],
              v73.m128i_i64[1],
              (__int64)v65,
              (__int64)v66,
              (const void *)v80.m128i_i64[0],
              v80.m128i_u64[1],
              v59,
              &v70);
      if ( (const __m128i **)v82.m128i_i64[0] != v83 )
        j_j___libc_free_0(v82.m128i_i64[0], &v83[0]->m128i_i8[1]);
LABEL_118:
      if ( v65 )
        j_j___libc_free_0(v65, (char *)v67 - (char *)v65);
      if ( (_QWORD *)v80.m128i_i64[0] != v81 )
        j_j___libc_free_0(v80.m128i_i64[0], v81[0] + 1LL);
LABEL_69:
      if ( (_QWORD *)v78.m128i_i64[0] != v79 )
        j_j___libc_free_0(v78.m128i_i64[0], v79[0] + 1LL);
LABEL_21:
      if ( v75 != v77 )
        j_j___libc_free_0(v75, v77[0] + 1LL);
      if ( (_QWORD *)v73.m128i_i64[0] != v74 )
        j_j___libc_free_0(v73.m128i_i64[0], v74[0] + 1LL);
      if ( v70 != v72 )
        j_j___libc_free_0(v70, v72[0] + 1LL);
      if ( (_QWORD *)v68.m128i_i64[0] != v69 )
        j_j___libc_free_0(v68.m128i_i64[0], v69[0] + 1LL);
      return v10;
    default:
LABEL_164:
      BUG();
  }
}

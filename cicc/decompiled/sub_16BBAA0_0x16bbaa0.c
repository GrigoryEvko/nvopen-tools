// Function: sub_16BBAA0
// Address: 0x16bbaa0
//
unsigned __int64 __fastcall sub_16BBAA0(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  const char *v4; // rdx
  __int64 v5; // rdx
  unsigned __int64 v6; // r8
  unsigned __int64 v7; // r13
  size_t v8; // r10
  const char *v9; // r13
  size_t v10; // rsi
  const char *v11; // rdi
  __int64 v12; // rdx
  const char *v13; // rbx
  size_t v14; // r13
  const char *v15; // rax
  const char *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rcx
  size_t v19; // r8
  __m128i *v20; // rdx
  __int64 v21; // r12
  __m128i v22; // xmm0
  __m128i *v23; // rdi
  unsigned __int64 v24; // rax
  __m128i v25; // xmm0
  __int64 v26; // rax
  __m128i *v27; // rdx
  __int64 v28; // rdi
  __m128i si128; // xmm0
  __int64 v30; // rax
  __m128i *v31; // rdx
  __int64 v32; // rdi
  __m128i v33; // xmm0
  __int64 v34; // rax
  size_t v35; // r10
  __m128i *v36; // rdx
  __m128i v37; // xmm0
  void *v38; // rdi
  __m128i *v39; // r10
  __int64 v40; // rdx
  const char *v41; // rsi
  __m128i v42; // xmm0
  const char *v43; // rdx
  __int64 v44; // rax
  const char *v45; // rdi
  const char **v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rdx
  int v49; // eax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  const char *v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rax
  int v56; // eax
  __int64 v57; // rax
  __m128i *v58; // rdx
  __int64 v59; // r12
  __m128i v60; // xmm0
  __m128i *v61; // rdi
  unsigned __int64 v62; // rax
  __m128i v63; // xmm0
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  char *v70; // rdi
  char *v71; // rdi
  __int64 v72; // rax
  unsigned __int64 v73; // [rsp-80h] [rbp-80h]
  size_t v74; // [rsp-78h] [rbp-78h]
  const char *v75; // [rsp-78h] [rbp-78h]
  const char *v76; // [rsp-78h] [rbp-78h]
  size_t v77; // [rsp-78h] [rbp-78h]
  const char *v78; // [rsp-70h] [rbp-70h]
  size_t v79; // [rsp-70h] [rbp-70h]
  __int64 v80; // [rsp-60h] [rbp-60h] BYREF
  const char *v81; // [rsp-58h] [rbp-58h] BYREF
  unsigned __int64 v82; // [rsp-50h] [rbp-50h]
  _QWORD v83[9]; // [rsp-48h] [rbp-48h] BYREF

  result = *(_QWORD *)(a2 + 8);
  if ( !result )
    return result;
  v4 = *(const char **)a2;
  v82 = *(_QWORD *)(a2 + 8);
  LOBYTE(v80) = 61;
  v81 = v4;
  v6 = sub_16D20C0(&v81, &v80, 1, 0);
  if ( v6 == -1 )
    goto LABEL_21;
  v7 = v6 + 1;
  v78 = v81;
  if ( v6 + 1 > v82 )
    v7 = v82;
  v8 = v82 - v7;
  v9 = &v81[v7];
  if ( v6 && v6 > v82 )
    v6 = v82;
  if ( !v8 )
  {
LABEL_21:
    v26 = sub_16E8CB0(&v81, &v80, v5);
    v27 = *(__m128i **)(v26 + 24);
    v28 = v26;
    if ( *(_QWORD *)(v26 + 16) - (_QWORD)v27 <= 0x13u )
    {
      v28 = sub_16E7EE0(v26, "DebugCounter Error: ", 20);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F66740);
      v27[1].m128i_i32[0] = 540701295;
      *v27 = si128;
      *(_QWORD *)(v26 + 24) += 20LL;
    }
    v30 = sub_16E7EE0(v28, *(const char **)a2, *(_QWORD *)(a2 + 8));
    v31 = *(__m128i **)(v30 + 24);
    v32 = v30;
    if ( *(_QWORD *)(v30 + 16) - (_QWORD)v31 > 0x19u )
    {
      v33 = _mm_load_si128((const __m128i *)&xmmword_3F66750);
      qmemcpy(&v31[1], "n = in it\n", 10);
      *v31 = v33;
      *(_QWORD *)(v30 + 24) += 26LL;
      return 0x69206E69203D206ELL;
    }
    v40 = 26;
    v41 = " does not have an = in it\n";
    return sub_16E7EE0(v32, v41, v40);
  }
  v10 = v8;
  v11 = v9;
  v73 = v6;
  v74 = v8;
  if ( (unsigned __int8)sub_16D2BB0(v9, v8, 0, &v81) )
  {
    v34 = sub_16E8CB0(v9, v10, v12);
    v35 = v74;
    v36 = *(__m128i **)(v34 + 24);
    v21 = v34;
    if ( *(_QWORD *)(v34 + 16) - (_QWORD)v36 <= 0x13u )
    {
      v67 = sub_16E7EE0(v34, "DebugCounter Error: ", 20);
      v35 = v74;
      v38 = *(void **)(v67 + 24);
      v21 = v67;
    }
    else
    {
      v37 = _mm_load_si128((const __m128i *)&xmmword_3F66740);
      v36[1].m128i_i32[0] = 540701295;
      *v36 = v37;
      v38 = (void *)(*(_QWORD *)(v34 + 24) + 20LL);
      *(_QWORD *)(v34 + 24) = v38;
    }
    if ( v35 > *(_QWORD *)(v21 + 16) - (_QWORD)v38 )
    {
      v50 = sub_16E7EE0(v21, v9, v35);
      v39 = *(__m128i **)(v50 + 24);
      v21 = v50;
    }
    else
    {
      v79 = v35;
      memcpy(v38, v9, v35);
      v39 = (__m128i *)(*(_QWORD *)(v21 + 24) + v79);
      *(_QWORD *)(v21 + 24) = v39;
    }
    v40 = 17;
    v41 = " is not a number\n";
    result = *(_QWORD *)(v21 + 16) - (_QWORD)v39;
    if ( result > 0x10 )
    {
      v42 = _mm_load_si128((const __m128i *)&xmmword_42AED30);
      v39[1].m128i_i8[0] = 10;
      *v39 = v42;
      *(_QWORD *)(v21 + 24) += 17LL;
      return result;
    }
    goto LABEL_46;
  }
  v13 = v81;
  if ( v73 <= 4 )
  {
LABEL_14:
    v17 = sub_16E8CB0(v11, v10, v12);
    v19 = v73;
    v20 = *(__m128i **)(v17 + 24);
    v21 = v17;
    if ( *(_QWORD *)(v17 + 16) - (_QWORD)v20 <= 0x13u )
    {
      v52 = sub_16E7EE0(v17, "DebugCounter Error: ", 20, v18, v73);
      v19 = v73;
      v23 = *(__m128i **)(v52 + 24);
      v21 = v52;
    }
    else
    {
      v22 = _mm_load_si128((const __m128i *)&xmmword_3F66740);
      v20[1].m128i_i32[0] = 540701295;
      *v20 = v22;
      v23 = (__m128i *)(*(_QWORD *)(v17 + 24) + 20LL);
      *(_QWORD *)(v17 + 24) = v23;
    }
    v24 = *(_QWORD *)(v21 + 16) - (_QWORD)v23;
    if ( v19 > v24 )
    {
      v51 = sub_16E7EE0(v21, v78, v19);
      v23 = *(__m128i **)(v51 + 24);
      v21 = v51;
      v24 = *(_QWORD *)(v51 + 16) - (_QWORD)v23;
    }
    else if ( v19 )
    {
      v77 = v19;
      memcpy(v23, v78, v19);
      v64 = *(_QWORD *)(v21 + 16);
      v23 = (__m128i *)(v77 + *(_QWORD *)(v21 + 24));
      *(_QWORD *)(v21 + 24) = v23;
      v24 = v64 - (_QWORD)v23;
    }
    if ( v24 > 0x22 )
    {
      v25 = _mm_load_si128((const __m128i *)&xmmword_42AED40);
      v23[2].m128i_i8[2] = 10;
      v23[2].m128i_i16[0] = 29806;
      *v23 = v25;
      v23[1] = _mm_load_si128((const __m128i *)&xmmword_42AED50);
      *(_QWORD *)(v21 + 24) += 35LL;
      return 29806;
    }
    v40 = 35;
    v41 = " does not end with -skip or -count\n";
LABEL_46:
    v32 = v21;
    return sub_16E7EE0(v32, v41, v40);
  }
  v14 = v73 - 5;
  v15 = &v78[v73 - 5];
  if ( *(_DWORD *)v15 != 1768649517 || v15[4] != 112 )
  {
    if ( v73 != 5 )
    {
      v14 = v73 - 6;
      v16 = &v78[v73 - 6];
      if ( *(_DWORD *)v16 == 1970234157 && *((_WORD *)v16 + 2) == 29806 )
      {
        if ( !v78 )
        {
          v45 = (const char *)(a1 + 32);
          v46 = &v81;
          LOBYTE(v83[0]) = 0;
          v81 = (const char *)v83;
          v82 = 0;
          v66 = sub_C61310(a1 + 32, (__int64)&v81);
          v48 = a1 + 40;
          if ( v66 == a1 + 40 )
          {
LABEL_59:
            LODWORD(v80) = 0;
            goto LABEL_60;
          }
          v49 = *(_DWORD *)(v66 + 64);
          LODWORD(v80) = v49;
          goto LABEL_42;
        }
        v80 = v73 - 6;
        v81 = (const char *)v83;
        if ( v14 > 0xF )
        {
          v81 = (const char *)sub_22409D0(&v81, &v80, 0);
          v71 = (char *)v81;
          v83[0] = v80;
        }
        else
        {
          if ( v73 == 7 )
          {
            v43 = (const char *)v83;
            LOBYTE(v83[0]) = *v78;
            v44 = 1;
LABEL_39:
            v82 = v44;
            v45 = (const char *)(a1 + 32);
            v46 = &v81;
            v43[v44] = 0;
            v75 = v81;
            v47 = sub_C61310(a1 + 32, (__int64)&v81);
            v48 = a1 + 40;
            if ( v47 == a1 + 40 )
            {
              LODWORD(v80) = 0;
              if ( v75 == (const char *)v83 )
                goto LABEL_60;
            }
            else
            {
              v49 = *(_DWORD *)(v47 + 64);
              LODWORD(v80) = v49;
              if ( v75 == (const char *)v83 )
                goto LABEL_42;
            }
            v45 = v75;
            v46 = (const char **)(v83[0] + 1LL);
            j_j___libc_free_0(v75, v83[0] + 1LL);
            v49 = v80;
LABEL_42:
            if ( v49 )
            {
              *(_BYTE *)(sub_16BAF20() + 104) = 1;
              *((_QWORD *)sub_16BB850(a1, (int *)&v80) + 3) = v13;
              result = (unsigned __int64)sub_16BB850(a1, (int *)&v80);
              *(_BYTE *)(result + 32) = 1;
              return result;
            }
            goto LABEL_60;
          }
          if ( v73 == 6 )
          {
            v44 = 0;
            v43 = (const char *)v83;
            goto LABEL_39;
          }
          v71 = (char *)v83;
        }
        memcpy(v71, v78, v14);
        v44 = v80;
        v43 = v81;
        goto LABEL_39;
      }
    }
    goto LABEL_14;
  }
  if ( !v78 )
  {
    v45 = (const char *)(a1 + 32);
    v46 = &v81;
    LOBYTE(v83[0]) = 0;
    v81 = (const char *)v83;
    v82 = 0;
    v65 = sub_C61310(a1 + 32, (__int64)&v81);
    v48 = a1 + 40;
    if ( v65 != a1 + 40 )
    {
      v56 = *(_DWORD *)(v65 + 64);
      LODWORD(v80) = v56;
LABEL_57:
      if ( v56 )
      {
        *(_BYTE *)(sub_16BAF20() + 104) = 1;
        *((_QWORD *)sub_16BB850(a1, (int *)&v80) + 2) = v13;
        result = (unsigned __int64)sub_16BB850(a1, (int *)&v80);
        *(_BYTE *)(result + 32) = 1;
        return result;
      }
      goto LABEL_60;
    }
    goto LABEL_59;
  }
  v80 = v73 - 5;
  v81 = (const char *)v83;
  if ( v14 > 0xF )
  {
    v81 = (const char *)sub_22409D0(&v81, &v80, 0);
    v70 = (char *)v81;
    v83[0] = v80;
    goto LABEL_78;
  }
  if ( v73 != 6 )
  {
    if ( v73 == 5 )
    {
      v54 = 0;
      v53 = (const char *)v83;
      goto LABEL_54;
    }
    v70 = (char *)v83;
LABEL_78:
    memcpy(v70, v78, v14);
    v54 = v80;
    v53 = v81;
    goto LABEL_54;
  }
  v53 = (const char *)v83;
  LOBYTE(v83[0]) = *v78;
  v54 = 1;
LABEL_54:
  v82 = v54;
  v45 = (const char *)(a1 + 32);
  v46 = &v81;
  v53[v54] = 0;
  v76 = v81;
  v55 = sub_C61310(a1 + 32, (__int64)&v81);
  v48 = a1 + 40;
  if ( v55 != a1 + 40 )
  {
    v56 = *(_DWORD *)(v55 + 64);
    LODWORD(v80) = v56;
    if ( v76 == (const char *)v83 )
      goto LABEL_57;
    goto LABEL_56;
  }
  LODWORD(v80) = 0;
  if ( v76 != (const char *)v83 )
  {
LABEL_56:
    v45 = v76;
    v46 = (const char **)(v83[0] + 1LL);
    j_j___libc_free_0(v76, v83[0] + 1LL);
    v56 = v80;
    goto LABEL_57;
  }
LABEL_60:
  v57 = sub_16E8CB0(v45, v46, v48);
  v58 = *(__m128i **)(v57 + 24);
  v59 = v57;
  if ( *(_QWORD *)(v57 + 16) - (_QWORD)v58 <= 0x13u )
  {
    v69 = sub_16E7EE0(v57, "DebugCounter Error: ", 20);
    v61 = *(__m128i **)(v69 + 24);
    v59 = v69;
  }
  else
  {
    v60 = _mm_load_si128((const __m128i *)&xmmword_3F66740);
    v58[1].m128i_i32[0] = 540701295;
    *v58 = v60;
    v61 = (__m128i *)(*(_QWORD *)(v57 + 24) + 20LL);
    *(_QWORD *)(v57 + 24) = v61;
  }
  v62 = *(_QWORD *)(v59 + 16) - (_QWORD)v61;
  if ( v14 > v62 )
  {
    v68 = sub_16E7EE0(v59, v78, v14);
    v61 = *(__m128i **)(v68 + 24);
    v59 = v68;
    v62 = *(_QWORD *)(v68 + 16) - (_QWORD)v61;
  }
  else if ( v14 )
  {
    memcpy(v61, v78, v14);
    v72 = *(_QWORD *)(v59 + 16);
    v61 = (__m128i *)(v14 + *(_QWORD *)(v59 + 24));
    *(_QWORD *)(v59 + 24) = v61;
    v62 = v72 - (_QWORD)v61;
  }
  if ( v62 <= 0x1C )
    return sub_16E7EE0(v59, " is not a registered counter\n", 29);
  v63 = _mm_load_si128((const __m128i *)&xmmword_3F66760);
  qmemcpy(&v61[1], "ered counter\n", 13);
  *v61 = v63;
  *(_QWORD *)(v59 + 24) += 29LL;
  return 0x756F632064657265LL;
}

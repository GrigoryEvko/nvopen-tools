// Function: sub_29F5D00
// Address: 0x29f5d00
//
void __fastcall sub_29F5D00(char *a1, __int64 a2, __int64 a3, __m128i *a4)
{
  char *v4; // r14
  __m128i *v5; // rbx
  char *v6; // r9
  __int64 i; // r14
  int v8; // eax
  const void *v9; // r11
  size_t v10; // rdx
  __int64 v11; // r10
  size_t v12; // rcx
  signed __int64 v13; // rax
  signed __int64 v14; // rax
  __int64 v15; // r12
  char *v16; // r15
  char *v17; // r13
  __m128i *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rbx
  size_t v21; // r13
  size_t v22; // rdx
  __m128i *v23; // rcx
  __m128i *v24; // r11
  __int64 v25; // rax
  size_t v26; // r15
  __int64 v27; // r13
  size_t v28; // r8
  __m128i *v29; // rcx
  __int64 v30; // r14
  size_t v31; // r12
  size_t v32; // rdx
  size_t v33; // r15
  char *v34; // r9
  int v35; // eax
  signed __int64 v36; // rax
  signed __int64 v37; // rax
  __m128i *v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rdx
  size_t v41; // r12
  __m128i *v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rax
  __m128i *v45; // rdi
  __int64 v46; // r12
  __int64 v47; // r13
  __m128i *v48; // rsi
  __int64 v49; // rdx
  size_t v50; // rdx
  __int64 v53; // [rsp+18h] [rbp-A8h]
  char *src; // [rsp+28h] [rbp-98h]
  size_t v56; // [rsp+30h] [rbp-90h]
  size_t v57; // [rsp+30h] [rbp-90h]
  size_t v58; // [rsp+38h] [rbp-88h]
  size_t v59; // [rsp+38h] [rbp-88h]
  size_t v60; // [rsp+38h] [rbp-88h]
  size_t v61; // [rsp+38h] [rbp-88h]
  size_t n; // [rsp+40h] [rbp-80h]
  size_t na; // [rsp+40h] [rbp-80h]
  size_t nb; // [rsp+40h] [rbp-80h]
  size_t nc; // [rsp+40h] [rbp-80h]
  void *s1; // [rsp+48h] [rbp-78h]
  void *s1a; // [rsp+48h] [rbp-78h]
  char *s1b; // [rsp+48h] [rbp-78h]
  __m128i *s1c; // [rsp+48h] [rbp-78h]
  __m128i *v70; // [rsp+50h] [rbp-70h]
  char *v71; // [rsp+50h] [rbp-70h]
  __m128i *v72; // [rsp+50h] [rbp-70h]
  size_t v73; // [rsp+50h] [rbp-70h]
  __m128i *v74; // [rsp+60h] [rbp-60h]
  size_t v75; // [rsp+68h] [rbp-58h]
  __m128i v76; // [rsp+70h] [rbp-50h] BYREF
  unsigned __int64 v77; // [rsp+80h] [rbp-40h]

  v4 = a1;
  v53 = (a3 - 1) / 2;
  v5 = (__m128i *)&a1[40 * a2];
  v70 = v5 + 1;
  if ( a2 >= v53 )
  {
    v23 = v5 + 1;
    v15 = a2;
    goto LABEL_34;
  }
  v6 = a1;
  for ( i = a2; ; i = v15 )
  {
    v15 = 2 * (i + 1);
    v20 = 80 * (i + 1);
    v11 = (__int64)&v6[v20 - 40];
    v5 = (__m128i *)&v6[v20];
    v21 = v5->m128i_u64[1];
    v12 = *(_QWORD *)(v11 + 8);
    v16 = (char *)v5->m128i_i64[0];
    v9 = *(const void **)v11;
    v10 = v12;
    if ( v21 <= v12 )
      v10 = v5->m128i_u64[1];
    if ( v10 )
    {
      src = v6;
      v56 = *(_QWORD *)(v11 + 8);
      v58 = v11;
      n = v10;
      s1 = *(void **)v11;
      v8 = memcmp((const void *)v5->m128i_i64[0], *(const void **)v11, v10);
      v9 = s1;
      v10 = n;
      v11 = v58;
      v12 = v56;
      v6 = src;
      if ( v8 )
      {
        if ( v8 >= 0 )
          goto LABEL_8;
        goto LABEL_13;
      }
      v13 = v21 - v56;
      if ( (__int64)(v21 - v56) >= 0x80000000LL )
        goto LABEL_8;
    }
    else
    {
      v13 = v21 - v12;
      if ( (__int64)(v21 - v12) >= 0x80000000LL )
        goto LABEL_9;
    }
    if ( v13 > (__int64)0xFFFFFFFF7FFFFFFFLL && (int)v13 >= 0 )
    {
      if ( v10 )
      {
LABEL_8:
        v59 = (size_t)v6;
        na = v12;
        s1a = (void *)v11;
        LODWORD(v14) = memcmp(v9, v16, v10);
        v11 = (__int64)s1a;
        v12 = na;
        v6 = (char *)v59;
        if ( (_DWORD)v14 )
          goto LABEL_11;
      }
LABEL_9:
      v14 = v12 - v21;
      if ( (__int64)(v12 - v21) >= 0x80000000LL )
        goto LABEL_12;
      if ( v14 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_14;
LABEL_11:
      if ( (int)v14 < 0 )
        goto LABEL_14;
LABEL_12:
      if ( v5[2].m128i_i64[0] >= *(_QWORD *)(v11 + 32) )
        goto LABEL_14;
    }
LABEL_13:
    --v15;
    v5 = (__m128i *)&v6[40 * v15];
    v16 = (char *)v5->m128i_i64[0];
LABEL_14:
    v17 = &v6[40 * i];
    v18 = *(__m128i **)v17;
    if ( &v5[1] == (__m128i *)v16 )
    {
      v22 = v5->m128i_u64[1];
      if ( v22 )
      {
        if ( v22 == 1 )
        {
          v18->m128i_i8[0] = v5[1].m128i_i8[0];
          v22 = v5->m128i_u64[1];
          v18 = *(__m128i **)v17;
        }
        else
        {
          v71 = v6;
          memcpy(v18, v16, v22);
          v22 = v5->m128i_u64[1];
          v18 = *(__m128i **)v17;
          v6 = v71;
        }
      }
      *((_QWORD *)v17 + 1) = v22;
      v18->m128i_i8[v22] = 0;
      v18 = (__m128i *)v5->m128i_i64[0];
    }
    else
    {
      if ( v18 == v70 )
      {
        *(_QWORD *)v17 = v16;
        *((_QWORD *)v17 + 1) = v5->m128i_i64[1];
        *((_QWORD *)v17 + 2) = v5[1].m128i_i64[0];
      }
      else
      {
        *(_QWORD *)v17 = v16;
        v19 = *((_QWORD *)v17 + 2);
        *((_QWORD *)v17 + 1) = v5->m128i_i64[1];
        *((_QWORD *)v17 + 2) = v5[1].m128i_i64[0];
        if ( v18 )
        {
          v5->m128i_i64[0] = (__int64)v18;
          v5[1].m128i_i64[0] = v19;
          goto LABEL_18;
        }
      }
      v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
      v18 = v5 + 1;
    }
LABEL_18:
    v5->m128i_i64[1] = 0;
    v18->m128i_i8[0] = 0;
    *((_QWORD *)v17 + 4) = v5[2].m128i_i64[0];
    if ( v15 >= v53 )
      break;
    v70 = v5 + 1;
  }
  v23 = v5 + 1;
  v4 = v6;
LABEL_34:
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v15 )
  {
    v44 = v15 + 1;
    v45 = (__m128i *)v5->m128i_i64[0];
    v46 = 2 * (v15 + 1);
    v47 = (__int64)&v4[64 * v44 - 40 + 8 * v46];
    v48 = *(__m128i **)v47;
    if ( *(_QWORD *)v47 == v47 + 16 )
    {
      v50 = *(_QWORD *)(v47 + 8);
      if ( v50 )
      {
        if ( v50 == 1 )
          v45->m128i_i8[0] = *(_BYTE *)(v47 + 16);
        else
          memcpy(v45, v48, v50);
        v50 = *(_QWORD *)(v47 + 8);
        v45 = (__m128i *)v5->m128i_i64[0];
      }
      v5->m128i_i64[1] = v50;
      v45->m128i_i8[v50] = 0;
      v45 = *(__m128i **)v47;
      goto LABEL_91;
    }
    if ( v45 == v23 )
    {
      v5->m128i_i64[0] = (__int64)v48;
      v5->m128i_i64[1] = *(_QWORD *)(v47 + 8);
      v5[1].m128i_i64[0] = *(_QWORD *)(v47 + 16);
    }
    else
    {
      v5->m128i_i64[0] = (__int64)v48;
      v49 = v5[1].m128i_i64[0];
      v5->m128i_i64[1] = *(_QWORD *)(v47 + 8);
      v5[1].m128i_i64[0] = *(_QWORD *)(v47 + 16);
      if ( v45 )
      {
        *(_QWORD *)v47 = v45;
        *(_QWORD *)(v47 + 16) = v49;
LABEL_91:
        *(_QWORD *)(v47 + 8) = 0;
        v15 = v46 - 1;
        v45->m128i_i8[0] = 0;
        v5[2].m128i_i64[0] = *(_QWORD *)(v47 + 32);
        v5 = (__m128i *)&v4[40 * v15];
        v23 = v5 + 1;
        goto LABEL_36;
      }
    }
    *(_QWORD *)v47 = v47 + 16;
    v45 = (__m128i *)(v47 + 16);
    goto LABEL_91;
  }
LABEL_36:
  v74 = &v76;
  v24 = (__m128i *)a4->m128i_i64[0];
  if ( (__m128i *)a4->m128i_i64[0] == &a4[1] )
  {
    v24 = &v76;
    v76 = _mm_loadu_si128(a4 + 1);
  }
  else
  {
    v74 = (__m128i *)a4->m128i_i64[0];
    v76.m128i_i64[0] = a4[1].m128i_i64[0];
  }
  a4->m128i_i64[0] = (__int64)a4[1].m128i_i64;
  v25 = a4[2].m128i_i64[0];
  v26 = a4->m128i_u64[1];
  a4[1].m128i_i8[0] = 0;
  v77 = v25;
  v75 = v26;
  a4->m128i_i64[1] = 0;
  v27 = (v15 - 1) / 2;
  if ( v15 <= a2 )
  {
LABEL_54:
    v38 = (__m128i *)v5->m128i_i64[0];
    if ( v24 == &v76 )
      goto LABEL_79;
    goto LABEL_55;
  }
  v72 = v23;
  v28 = (size_t)v4;
  v29 = v24;
  v30 = v15;
  v31 = v26;
  while ( 2 )
  {
    v32 = v31;
    v5 = (__m128i *)(v28 + 40 * v27);
    v33 = v5->m128i_u64[1];
    v34 = (char *)v5->m128i_i64[0];
    if ( v33 <= v31 )
      v32 = v5->m128i_u64[1];
    if ( !v32 )
    {
      v36 = v33 - v31;
      if ( (__int64)(v33 - v31) < 0x80000000LL )
        goto LABEL_45;
      goto LABEL_49;
    }
    nb = (size_t)v29;
    v57 = v28;
    v60 = v32;
    s1b = (char *)v5->m128i_i64[0];
    v35 = memcmp((const void *)v5->m128i_i64[0], v29, v32);
    v34 = s1b;
    v29 = (__m128i *)nb;
    v32 = v60;
    v28 = v57;
    if ( v35 )
    {
      if ( v35 >= 0 )
      {
LABEL_48:
        v61 = v28;
        nc = (size_t)v34;
        s1c = v29;
        LODWORD(v37) = memcmp(v29, v34, v32);
        v29 = s1c;
        v34 = (char *)nc;
        v28 = v61;
        if ( !(_DWORD)v37 )
          goto LABEL_49;
LABEL_51:
        if ( (int)v37 < 0 )
          goto LABEL_53;
LABEL_52:
        if ( v5[2].m128i_i64[0] >= v77 )
        {
LABEL_53:
          v24 = v29;
          v23 = v72;
          v26 = v31;
          v5 = (__m128i *)(v28 + 40 * v30);
          goto LABEL_54;
        }
      }
    }
    else
    {
      v36 = v33 - v31;
      if ( (__int64)(v33 - v31) >= 0x80000000LL )
        goto LABEL_48;
LABEL_45:
      if ( v36 > (__int64)0xFFFFFFFF7FFFFFFFLL && (int)v36 >= 0 )
      {
        if ( v32 )
          goto LABEL_48;
LABEL_49:
        v37 = v31 - v33;
        if ( (__int64)(v31 - v33) < 0x80000000LL )
        {
          if ( v37 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
            goto LABEL_53;
          goto LABEL_51;
        }
        goto LABEL_52;
      }
    }
    v41 = v28 + 40 * v30;
    v42 = *(__m128i **)v41;
    if ( v34 == (char *)&v5[1] )
    {
      if ( v33 )
      {
        if ( v33 == 1 )
        {
          v42->m128i_i8[0] = v5[1].m128i_i8[0];
          v33 = v5->m128i_u64[1];
          v42 = *(__m128i **)v41;
        }
        else
        {
          v73 = v28;
          memcpy(v42, &v5[1], v33);
          v33 = v5->m128i_u64[1];
          v42 = *(__m128i **)v41;
          v28 = v73;
        }
      }
      *(_QWORD *)(v41 + 8) = v33;
      v42->m128i_i8[v33] = 0;
      v42 = (__m128i *)v5->m128i_i64[0];
    }
    else
    {
      if ( v42 == v72 )
      {
        *(_QWORD *)v41 = v34;
        *(_QWORD *)(v41 + 8) = v5->m128i_i64[1];
        *(_QWORD *)(v41 + 16) = v5[1].m128i_i64[0];
      }
      else
      {
        *(_QWORD *)v41 = v34;
        v43 = *(_QWORD *)(v41 + 16);
        *(_QWORD *)(v41 + 8) = v5->m128i_i64[1];
        *(_QWORD *)(v41 + 16) = v5[1].m128i_i64[0];
        if ( v42 )
        {
          v5->m128i_i64[0] = (__int64)v42;
          v5[1].m128i_i64[0] = v43;
          goto LABEL_66;
        }
      }
      v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
      v42 = v5 + 1;
    }
LABEL_66:
    v5->m128i_i64[1] = 0;
    v42->m128i_i8[0] = 0;
    *(_QWORD *)(v41 + 32) = v5[2].m128i_i64[0];
    if ( v27 > a2 )
    {
      v29 = v74;
      v31 = v75;
      v72 = v5 + 1;
      v30 = v27;
      v27 = (v27 - 1) / 2;
      continue;
    }
    break;
  }
  v26 = v75;
  v24 = v74;
  v23 = v5 + 1;
  v38 = (__m128i *)v5->m128i_i64[0];
  if ( v74 == &v76 )
  {
LABEL_79:
    if ( v26 )
    {
      if ( v26 == 1 )
        v38->m128i_i8[0] = v76.m128i_i8[0];
      else
        memcpy(v38, &v76, v26);
      v26 = v75;
      v38 = (__m128i *)v5->m128i_i64[0];
    }
    v5->m128i_i64[1] = v26;
    v38->m128i_i8[v26] = 0;
    v38 = v74;
    goto LABEL_58;
  }
LABEL_55:
  v39 = v76.m128i_i64[0];
  if ( v38 == v23 )
  {
    v5->m128i_i64[0] = (__int64)v24;
    v5->m128i_i64[1] = v26;
    v5[1].m128i_i64[0] = v39;
  }
  else
  {
    v40 = v5[1].m128i_i64[0];
    v5->m128i_i64[0] = (__int64)v24;
    v5->m128i_i64[1] = v26;
    v5[1].m128i_i64[0] = v39;
    if ( v38 )
    {
      v74 = v38;
      v76.m128i_i64[0] = v40;
      goto LABEL_58;
    }
  }
  v74 = &v76;
  v38 = &v76;
LABEL_58:
  v38->m128i_i8[0] = 0;
  v5[2].m128i_i64[0] = v77;
  if ( v74 != &v76 )
    j_j___libc_free_0((unsigned __int64)v74);
}

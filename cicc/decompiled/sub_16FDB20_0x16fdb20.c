// Function: sub_16FDB20
// Address: 0x16fdb20
//
__m128i *__fastcall sub_16FDB20(__m128i *a1, __int64 a2)
{
  unsigned __int64 v3; // rbx
  _BYTE *v4; // r14
  size_t v5; // r12
  __int64 *v6; // rax
  __int64 v7; // rdi
  _QWORD *v8; // rax
  size_t v9; // rcx
  _BYTE *v10; // r9
  size_t v11; // r12
  size_t *v12; // rax
  _BYTE *v13; // rdi
  size_t *v14; // rax
  __int64 v15; // rdi
  void *v16; // rax
  unsigned int v17; // eax
  __m128i *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __m128i v21; // xmm0
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __m128i v27; // xmm0
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 v30; // r12
  __int64 v31; // r12
  _QWORD *v32; // rax
  size_t v33; // rcx
  __int64 v34; // r8
  _BYTE *v35; // r9
  size_t v36; // r12
  size_t *v37; // rax
  _BYTE *v38; // rdi
  size_t *v39; // rax
  __int64 v40; // rdi
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __int64 v43; // rdi
  _QWORD *v44; // rax
  size_t v45; // rcx
  _BYTE *v46; // r9
  size_t v47; // r12
  size_t *v48; // rax
  _BYTE *v49; // rdi
  size_t *v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rdx
  __m128i v54; // xmm0
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  __m128i si128; // xmm0
  __int64 v60; // rax
  __int64 v61; // rdx
  __m128i v62; // xmm1
  size_t v63; // rdx
  size_t v64; // rdx
  __int64 v65; // rax
  size_t *v66; // rdi
  __int64 v67; // rax
  size_t *v68; // rdi
  __int64 v69; // rax
  size_t *v70; // rdi
  _BYTE *src; // [rsp+0h] [rbp-D0h]
  _BYTE *srca; // [rsp+0h] [rbp-D0h]
  _BYTE *srcb; // [rsp+0h] [rbp-D0h]
  __m128i v74; // [rsp+10h] [rbp-C0h] BYREF
  const char *v75; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v76; // [rsp+28h] [rbp-A8h]
  __int16 v77; // [rsp+30h] [rbp-A0h]
  void *dest; // [rsp+40h] [rbp-90h] BYREF
  size_t v79; // [rsp+48h] [rbp-88h]
  __m128i v80; // [rsp+50h] [rbp-80h] BYREF
  __int64 v81; // [rsp+60h] [rbp-70h] BYREF
  size_t n[2]; // [rsp+68h] [rbp-68h] BYREF
  _QWORD *v83; // [rsp+78h] [rbp-58h]
  __int64 v84; // [rsp+80h] [rbp-50h]
  _QWORD v85[9]; // [rsp+88h] [rbp-48h] BYREF

  v3 = *(_QWORD *)(a2 + 64);
  if ( !v3 || (v4 = *(_BYTE **)(a2 + 56), v3 == 1) && *v4 == 33 )
  {
    v17 = *(_DWORD *)(a2 + 32);
    v18 = a1 + 1;
    if ( v17 == 4 )
    {
      a1->m128i_i64[0] = (__int64)v18;
      v81 = 21;
      v57 = sub_22409D0(a1, &v81, 0);
      v58 = v81;
      si128 = _mm_load_si128((const __m128i *)&xmmword_42B0030);
      a1->m128i_i64[0] = v57;
      a1[1].m128i_i64[0] = v58;
      *(_DWORD *)(v57 + 16) = 1634548274;
      *(_BYTE *)(v57 + 20) = 112;
      *(__m128i *)v57 = si128;
      v60 = v81;
      v61 = a1->m128i_i64[0];
      a1->m128i_i64[1] = v81;
      *(_BYTE *)(v61 + v60) = 0;
      return a1;
    }
    if ( v17 > 4 )
    {
      if ( v17 == 5 )
      {
        a1->m128i_i64[0] = (__int64)v18;
        v81 = 21;
        v25 = sub_22409D0(a1, &v81, 0);
        v26 = v81;
        v27 = _mm_load_si128((const __m128i *)&xmmword_42B0030);
        a1->m128i_i64[0] = v25;
        a1[1].m128i_i64[0] = v26;
        *(_DWORD *)(v25 + 16) = 1702050354;
        *(_BYTE *)(v25 + 20) = 113;
        *(__m128i *)v25 = v27;
        v28 = v81;
        v29 = a1->m128i_i64[0];
        a1->m128i_i64[1] = v81;
        *(_BYTE *)(v29 + v28) = 0;
        return a1;
      }
    }
    else
    {
      if ( !v17 )
      {
        a1->m128i_i64[0] = (__int64)v18;
        v81 = 22;
        v52 = sub_22409D0(a1, &v81, 0);
        v53 = v81;
        v54 = _mm_load_si128((const __m128i *)&xmmword_42B0030);
        a1->m128i_i64[0] = v52;
        a1[1].m128i_i64[0] = v53;
        *(_DWORD *)(v52 + 16) = 1970158130;
        *(_WORD *)(v52 + 20) = 27756;
        *(__m128i *)v52 = v54;
        v55 = v81;
        v56 = a1->m128i_i64[0];
        a1->m128i_i64[1] = v81;
        *(_BYTE *)(v56 + v55) = 0;
        return a1;
      }
      if ( v17 != 3 )
      {
        a1->m128i_i64[0] = (__int64)v18;
        v81 = 21;
        v19 = sub_22409D0(a1, &v81, 0);
        v20 = v81;
        v21 = _mm_load_si128((const __m128i *)&xmmword_42B0030);
        a1->m128i_i64[0] = v19;
        a1[1].m128i_i64[0] = v20;
        *(_DWORD *)(v19 + 16) = 1953708594;
        *(_BYTE *)(v19 + 20) = 114;
        *(__m128i *)v19 = v21;
        v22 = v81;
        v23 = a1->m128i_i64[0];
        a1->m128i_i64[1] = v81;
        *(_BYTE *)(v23 + v22) = 0;
        return a1;
      }
    }
    a1->m128i_i64[0] = (__int64)v18;
    a1->m128i_i64[1] = 0;
    a1[1].m128i_i8[0] = 0;
    return a1;
  }
  v80.m128i_i8[0] = 0;
  v5 = v3;
  dest = &v80;
  v79 = 0;
  do
  {
    if ( v4[--v5] == 33 )
    {
      if ( v5 )
        goto LABEL_7;
      v43 = **(_QWORD **)(a2 + 8);
      v76 = 1;
      v75 = "!";
      v44 = sub_16FDA10(v43 + 120, (__int64)&v75);
      v46 = (_BYTE *)v44[6];
      if ( !v46 )
      {
        LOBYTE(n[1]) = 0;
        v49 = dest;
        v81 = (__int64)&n[1];
        goto LABEL_75;
      }
      v47 = v44[7];
      v81 = (__int64)&n[1];
      v74.m128i_i64[0] = v47;
      if ( v47 > 0xF )
      {
        src = v46;
        v65 = sub_22409D0(&v81, &v74, 0);
        v46 = src;
        v81 = v65;
        v66 = (size_t *)v65;
        n[1] = v74.m128i_i64[0];
      }
      else
      {
        if ( v47 == 1 )
        {
          LOBYTE(n[1]) = *v46;
          v48 = &n[1];
LABEL_56:
          n[0] = v47;
          *((_BYTE *)v48 + v47) = 0;
          v49 = dest;
          v50 = (size_t *)dest;
          if ( (size_t *)v81 != &n[1] )
          {
            v45 = n[1];
            if ( dest == &v80 )
            {
              dest = (void *)v81;
              v79 = n[0];
              v80.m128i_i64[0] = n[1];
            }
            else
            {
              v51 = v80.m128i_i64[0];
              dest = (void *)v81;
              v79 = n[0];
              v80.m128i_i64[0] = n[1];
              if ( v50 )
              {
                v81 = (__int64)v50;
                n[1] = v51;
LABEL_60:
                n[0] = 0;
                *(_BYTE *)v50 = 0;
                if ( (size_t *)v81 != &n[1] )
                  j_j___libc_free_0(v81, n[1] + 1);
                if ( 0x3FFFFFFFFFFFFFFFLL - v79 < v3 - 1 )
                  goto LABEL_115;
                sub_2241490(&dest, v4 + 1, v3 - 1, v45);
                a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
                v16 = dest;
                if ( dest != &v80 )
                  goto LABEL_49;
                goto LABEL_21;
              }
            }
            v81 = (__int64)&n[1];
            v50 = &n[1];
            goto LABEL_60;
          }
          v5 = n[0];
          if ( n[0] )
          {
            if ( n[0] == 1 )
              *(_BYTE *)dest = n[1];
            else
              memcpy(dest, &n[1], n[0]);
            v5 = n[0];
            v49 = dest;
          }
LABEL_75:
          v79 = v5;
          v49[v5] = 0;
          v50 = (size_t *)v81;
          goto LABEL_60;
        }
        if ( !v47 )
        {
          v48 = &n[1];
          goto LABEL_56;
        }
        v66 = &n[1];
      }
      memcpy(v66, v46, v47);
      v47 = v74.m128i_i64[0];
      v48 = (size_t *)v81;
      goto LABEL_56;
    }
  }
  while ( v5 );
  v5 = -1;
LABEL_7:
  v6 = *(__int64 **)(a2 + 8);
  if ( v3 == 1 || *(_WORD *)v4 != 8481 )
  {
    v30 = v5 + 1;
    v74.m128i_i64[0] = (__int64)v4;
    if ( v30 > v3 )
      v30 = v3;
    v74.m128i_i64[1] = v30;
    v31 = *v6 + 128;
    v32 = sub_16FDA10(*v6 + 120, (__int64)&v74);
    if ( v32 == (_QWORD *)v31 )
    {
      LOBYTE(v85[0]) = 0;
      v62 = _mm_load_si128(&v74);
      v76 = (__int64)&v74;
      v83 = v85;
      v84 = 0;
      LODWORD(v81) = 22;
      v75 = "Unknown tag handle ";
      v77 = 1283;
      *(__m128i *)n = v62;
      sub_16F8380(a2, (__int64)&v75, (__int64)&v81, 1283, v34);
      if ( v83 != v85 )
        j_j___libc_free_0(v83, v85[0] + 1LL);
LABEL_44:
      v41 = v3;
      while ( 1 )
      {
        v42 = v41--;
        if ( v4[v41] == 33 )
          break;
        if ( !v41 )
          goto LABEL_47;
      }
      if ( v3 >= v42 )
      {
        v3 -= v42;
        v4 += v42;
LABEL_47:
        if ( v3 > 0x3FFFFFFFFFFFFFFFLL - v79 )
          goto LABEL_115;
      }
      else
      {
        v4 += v3;
        v3 = 0;
      }
      sub_2241490(&dest, v4, v3, v33);
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      v16 = dest;
      if ( dest != &v80 )
        goto LABEL_49;
LABEL_21:
      a1[1] = _mm_load_si128(&v80);
      goto LABEL_50;
    }
    v35 = (_BYTE *)v32[6];
    if ( !v35 )
    {
      LOBYTE(n[1]) = 0;
      v38 = dest;
      v63 = 0;
      v81 = (__int64)&n[1];
      goto LABEL_73;
    }
    v36 = v32[7];
    v81 = (__int64)&n[1];
    v75 = (const char *)v36;
    if ( v36 > 0xF )
    {
      srca = v35;
      v67 = sub_22409D0(&v81, &v75, 0);
      v35 = srca;
      v81 = v67;
      v68 = (size_t *)v67;
      n[1] = (size_t)v75;
    }
    else
    {
      if ( v36 == 1 )
      {
        LOBYTE(n[1]) = *v35;
        v37 = &n[1];
LABEL_38:
        n[0] = v36;
        *((_BYTE *)v37 + v36) = 0;
        v38 = dest;
        v39 = (size_t *)dest;
        if ( (size_t *)v81 != &n[1] )
        {
          v33 = n[1];
          if ( dest == &v80 )
          {
            dest = (void *)v81;
            v79 = n[0];
            v80.m128i_i64[0] = n[1];
          }
          else
          {
            v40 = v80.m128i_i64[0];
            dest = (void *)v81;
            v79 = n[0];
            v80.m128i_i64[0] = n[1];
            if ( v39 )
            {
              v81 = (__int64)v39;
              n[1] = v40;
              goto LABEL_42;
            }
          }
          v81 = (__int64)&n[1];
          v39 = &n[1];
LABEL_42:
          n[0] = 0;
          *(_BYTE *)v39 = 0;
          if ( (size_t *)v81 != &n[1] )
            j_j___libc_free_0(v81, n[1] + 1);
          goto LABEL_44;
        }
        v63 = n[0];
        if ( n[0] )
        {
          if ( n[0] == 1 )
            *(_BYTE *)dest = n[1];
          else
            memcpy(dest, &n[1], n[0]);
          v63 = n[0];
          v38 = dest;
        }
LABEL_73:
        v79 = v63;
        v38[v63] = 0;
        v39 = (size_t *)v81;
        goto LABEL_42;
      }
      if ( !v36 )
      {
        v37 = &n[1];
        goto LABEL_38;
      }
      v68 = &n[1];
    }
    memcpy(v68, v35, v36);
    v36 = (size_t)v75;
    v37 = (size_t *)v81;
    goto LABEL_38;
  }
  v7 = *v6;
  v76 = 2;
  v75 = (const char *)&unk_3F6A4C4;
  v8 = sub_16FDA10(v7 + 120, (__int64)&v75);
  v10 = (_BYTE *)v8[6];
  if ( !v10 )
  {
    LOBYTE(n[1]) = 0;
    v13 = dest;
    v64 = 0;
    v81 = (__int64)&n[1];
LABEL_78:
    v79 = v64;
    v13[v64] = 0;
    v14 = (size_t *)v81;
    goto LABEL_17;
  }
  v11 = v8[7];
  v81 = (__int64)&n[1];
  v74.m128i_i64[0] = v11;
  if ( v11 > 0xF )
  {
    srcb = v10;
    v69 = sub_22409D0(&v81, &v74, 0);
    v10 = srcb;
    v81 = v69;
    v70 = (size_t *)v69;
    n[1] = v74.m128i_i64[0];
  }
  else
  {
    if ( v11 == 1 )
    {
      LOBYTE(n[1]) = *v10;
      v12 = &n[1];
      goto LABEL_13;
    }
    if ( !v11 )
    {
      v12 = &n[1];
      goto LABEL_13;
    }
    v70 = &n[1];
  }
  memcpy(v70, v10, v11);
  v11 = v74.m128i_i64[0];
  v12 = (size_t *)v81;
LABEL_13:
  n[0] = v11;
  *((_BYTE *)v12 + v11) = 0;
  v13 = dest;
  v14 = (size_t *)dest;
  if ( (size_t *)v81 == &n[1] )
  {
    v64 = n[0];
    if ( n[0] )
    {
      if ( n[0] == 1 )
        *(_BYTE *)dest = n[1];
      else
        memcpy(dest, &n[1], n[0]);
      v64 = n[0];
      v13 = dest;
    }
    goto LABEL_78;
  }
  v9 = n[1];
  if ( dest == &v80 )
  {
    dest = (void *)v81;
    v79 = n[0];
    v80.m128i_i64[0] = n[1];
  }
  else
  {
    v15 = v80.m128i_i64[0];
    dest = (void *)v81;
    v79 = n[0];
    v80.m128i_i64[0] = n[1];
    if ( v14 )
    {
      v81 = (__int64)v14;
      n[1] = v15;
      goto LABEL_17;
    }
  }
  v81 = (__int64)&n[1];
  v14 = &n[1];
LABEL_17:
  n[0] = 0;
  *(_BYTE *)v14 = 0;
  if ( (size_t *)v81 != &n[1] )
    j_j___libc_free_0(v81, n[1] + 1);
  if ( 0x3FFFFFFFFFFFFFFFLL - v79 < v3 - 2 )
LABEL_115:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&dest, v4 + 2, v3 - 2, v9);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  v16 = dest;
  if ( dest == &v80 )
    goto LABEL_21;
LABEL_49:
  a1->m128i_i64[0] = (__int64)v16;
  a1[1].m128i_i64[0] = v80.m128i_i64[0];
LABEL_50:
  a1->m128i_i64[1] = v79;
  return a1;
}

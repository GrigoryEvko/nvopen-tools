// Function: sub_273ACE0
// Address: 0x273ace0
//
char __fastcall sub_273ACE0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6, __int64 a7)
{
  __int64 v7; // rax
  __m128i *v9; // r13
  const __m128i *v10; // r12
  __m128i *v11; // rbx
  __int64 v12; // r15
  __int64 v13; // r9
  __int64 v14; // rbx
  __int64 v15; // r11
  __m128i *v16; // r12
  __int64 v17; // rax
  const __m128i *v18; // r9
  int v19; // r11d
  __int64 v20; // rcx
  __int64 v21; // r14
  const __m128i *v22; // r10
  size_t v23; // r10
  __int64 *v24; // r15
  __m128i v25; // xmm7
  unsigned __int32 v26; // eax
  __int32 v27; // eax
  int v28; // ecx
  __m128i v29; // xmm6
  __int64 v30; // r8
  __int64 v31; // rsi
  size_t v32; // rdx
  __int64 *v33; // r15
  const __m128i *v34; // r12
  __int64 *v35; // r15
  unsigned int v36; // eax
  unsigned __int64 v37; // rax
  __int64 v38; // r8
  unsigned __int64 v39; // rax
  int v40; // ecx
  int v41; // eax
  __int32 v42; // ecx
  unsigned __int64 v43; // r8
  unsigned __int64 v44; // rsi
  __int64 *v45; // rsi
  __m128i *v46; // rdi
  bool v47; // al
  __int64 v48; // rax
  size_t v49; // r8
  const __m128i *v50; // rax
  const __m128i *v51; // rax
  int v53; // [rsp+8h] [rbp-68h]
  const __m128i *v54; // [rsp+8h] [rbp-68h]
  int v55; // [rsp+10h] [rbp-60h]
  int v56; // [rsp+10h] [rbp-60h]
  int v57; // [rsp+10h] [rbp-60h]
  int v58; // [rsp+10h] [rbp-60h]
  int v59; // [rsp+10h] [rbp-60h]
  int v60; // [rsp+18h] [rbp-58h]
  void *v61; // [rsp+18h] [rbp-58h]
  void *v62; // [rsp+18h] [rbp-58h]
  int v63; // [rsp+18h] [rbp-58h]
  const __m128i *v64; // [rsp+18h] [rbp-58h]
  int v65; // [rsp+18h] [rbp-58h]
  void *v66; // [rsp+18h] [rbp-58h]
  int v67; // [rsp+18h] [rbp-58h]
  int v68; // [rsp+18h] [rbp-58h]
  const __m128i *srca; // [rsp+28h] [rbp-48h]
  const __m128i *srcb; // [rsp+28h] [rbp-48h]
  int srcc; // [rsp+28h] [rbp-48h]
  void *srcd; // [rsp+28h] [rbp-48h]
  __m128i *src; // [rsp+28h] [rbp-48h]
  void *srce; // [rsp+28h] [rbp-48h]
  void *srcf; // [rsp+28h] [rbp-48h]
  int srcg; // [rsp+28h] [rbp-48h]
  int srch; // [rsp+28h] [rbp-48h]
  int srci; // [rsp+28h] [rbp-48h]
  __int64 *dest; // [rsp+30h] [rbp-40h]
  const __m128i *v81; // [rsp+38h] [rbp-38h]

  v7 = a5;
  v9 = a1;
  v10 = (const __m128i *)a2;
  v11 = (__m128i *)a3;
  if ( a7 <= a5 )
    v7 = a7;
  if ( a4 <= v7 )
  {
LABEL_22:
    if ( v9 != v10 )
      memmove(a6, v9, (char *)v10 - (char *)v9);
    v24 = (__int64 *)((char *)a6 + (char *)v10 - (char *)v9);
    LOBYTE(v7) = v11 == v10;
    if ( v11 == v10 || v24 == a6 )
    {
LABEL_71:
      if ( v24 == a6 )
        return v7;
      v45 = a6;
      v46 = v9;
      v32 = (char *)v24 - (char *)a6;
LABEL_73:
      LOBYTE(v7) = (unsigned __int8)memmove(v46, v45, v32);
      return v7;
    }
    while ( 1 )
    {
      v26 = *((_DWORD *)a6 + 12);
      if ( v10[3].m128i_i32[0] != v26 )
      {
        if ( v10[3].m128i_i32[0] >= v26 )
          goto LABEL_27;
        goto LABEL_32;
      }
      v27 = v10[3].m128i_i32[2];
      v28 = *((_DWORD *)a6 + 14);
      if ( v27 )
      {
        if ( !v28 )
          goto LABEL_27;
        v30 = v10->m128i_i64[0];
        if ( v27 == 3 )
        {
          v39 = sub_2739680(v10->m128i_i64[0]);
          v31 = *a6;
          v30 = v39;
          if ( v40 != 3 )
          {
LABEL_37:
            if ( !sub_B445A0(v30, v31) )
              goto LABEL_27;
            goto LABEL_32;
          }
        }
        else
        {
          v31 = *a6;
          if ( v28 != 3 )
            goto LABEL_37;
        }
        v37 = sub_2739680(v31);
        if ( !sub_B445A0(v38, v37) )
          goto LABEL_27;
      }
      else if ( !v28 )
      {
        v47 = 0;
        if ( *(_BYTE *)v10->m128i_i64[1] != 17 )
          v47 = *(_BYTE *)v10[1].m128i_i64[0] != 17;
        if ( *(_BYTE *)a6[1] == 17 || (unsigned __int8)v47 >= (unsigned __int8)(*(_BYTE *)a6[2] != 17) )
        {
LABEL_27:
          v25 = _mm_loadu_si128((const __m128i *)a6);
          a6 += 8;
          v9 += 4;
          v9[-4] = v25;
          v9[-3] = _mm_loadu_si128((const __m128i *)a6 - 3);
          v9[-2] = _mm_loadu_si128((const __m128i *)a6 - 2);
          v9[-1].m128i_i64[0] = *(a6 - 2);
          LODWORD(v7) = *((_DWORD *)a6 - 2);
          v9[-1].m128i_i32[2] = v7;
          if ( v24 == a6 )
            return v7;
          goto LABEL_28;
        }
      }
LABEL_32:
      v29 = _mm_loadu_si128(v10);
      v9 += 4;
      v10 += 4;
      v9[-4] = v29;
      v9[-3] = _mm_loadu_si128(v10 - 3);
      v9[-2] = _mm_loadu_si128(v10 - 2);
      v9[-1].m128i_i64[0] = v10[-1].m128i_i64[0];
      LODWORD(v7) = v10[-1].m128i_i32[2];
      v9[-1].m128i_i32[2] = v7;
      if ( v24 == a6 )
        return v7;
LABEL_28:
      if ( v11 == v10 )
        goto LABEL_71;
    }
  }
  v12 = a5;
  if ( a7 >= a5 )
    goto LABEL_41;
  v13 = a2;
  v14 = a4;
  v15 = (__int64)a1;
  dest = a6;
  while ( 1 )
  {
    if ( v14 <= v12 )
    {
      v64 = (const __m128i *)v13;
      srce = (void *)v15;
      v21 = v12 / 2;
      v81 = (const __m128i *)(v13 + ((v12 / 2) << 6));
      v48 = sub_2739EE0(v15, v13, (__int64)v81);
      v19 = (int)srce;
      v18 = v64;
      v16 = (__m128i *)v48;
      v20 = (v48 - (__int64)srce) >> 6;
    }
    else
    {
      v60 = v15;
      srca = (const __m128i *)v13;
      v16 = (__m128i *)(v15 + ((v14 / 2) << 6));
      v17 = sub_2739FD0(v13, a3, (__int64)v16);
      v18 = srca;
      v19 = v60;
      v81 = (const __m128i *)v17;
      v20 = v14 / 2;
      v21 = (v17 - (__int64)srca) >> 6;
    }
    v14 -= v20;
    if ( v14 <= v21 || v21 > a7 )
    {
      if ( v14 > a7 )
      {
        v68 = v19;
        srci = v20;
        v51 = sub_27396F0(v16, v18, v81);
        v19 = v68;
        LODWORD(v20) = srci;
        v22 = v51;
      }
      else
      {
        v22 = v81;
        if ( v14 )
        {
          v49 = (char *)v18 - (char *)v16;
          if ( v18 != v16 )
          {
            v54 = v18;
            v58 = v19;
            v65 = v20;
            srcf = (void *)((char *)v18 - (char *)v16);
            memmove(dest, v16, (char *)v18 - (char *)v16);
            v18 = v54;
            v19 = v58;
            LODWORD(v20) = v65;
            v49 = (size_t)srcf;
          }
          if ( v18 != v81 )
          {
            v59 = v19;
            v66 = (void *)v49;
            srcg = v20;
            memmove(v16, v18, (char *)v81 - (char *)v18);
            v19 = v59;
            v49 = (size_t)v66;
            LODWORD(v20) = srcg;
          }
          v22 = (const __m128i *)((char *)v81 - v49);
          if ( v49 )
          {
            v67 = v19;
            srch = v20;
            v50 = (const __m128i *)memmove((char *)v81 - v49, dest, v49);
            LODWORD(v20) = srch;
            v19 = v67;
            v22 = v50;
          }
        }
      }
    }
    else
    {
      v22 = v16;
      if ( v21 )
      {
        v23 = (char *)v81 - (char *)v18;
        if ( v18 != v81 )
        {
          v53 = v19;
          v55 = v20;
          v61 = (void *)((char *)v81 - (char *)v18);
          srcb = v18;
          memmove(dest, v18, (char *)v81 - (char *)v18);
          v19 = v53;
          LODWORD(v20) = v55;
          v23 = (size_t)v61;
          v18 = srcb;
        }
        if ( v18 != v16 )
        {
          v56 = v19;
          v62 = (void *)v23;
          srcc = v20;
          memmove((char *)v81 - ((char *)v18 - (char *)v16), v16, (char *)v18 - (char *)v16);
          v19 = v56;
          v23 = (size_t)v62;
          LODWORD(v20) = srcc;
        }
        if ( v23 )
        {
          v57 = v19;
          v63 = v20;
          srcd = (void *)v23;
          memmove(v16, dest, v23);
          v19 = v57;
          LODWORD(v20) = v63;
          v23 = (size_t)srcd;
        }
        v22 = (__m128i *)((char *)v16 + v23);
      }
    }
    v12 -= v21;
    src = (__m128i *)v22;
    sub_273ACE0(v19, (_DWORD)v16, (_DWORD)v22, v20, v21, (_DWORD)dest, a7);
    v7 = a7;
    if ( v12 <= a7 )
      v7 = v12;
    if ( v14 <= v7 )
    {
      v11 = (__m128i *)a3;
      a6 = dest;
      v9 = src;
      v10 = v81;
      goto LABEL_22;
    }
    if ( v12 <= a7 )
      break;
    v13 = (__int64)v81;
    v15 = (__int64)src;
  }
  v11 = (__m128i *)a3;
  a6 = dest;
  v9 = src;
  v10 = v81;
LABEL_41:
  v32 = (char *)v11 - (char *)v10;
  if ( v11 != v10 )
  {
    LOBYTE(v7) = (unsigned __int8)memmove(a6, v10, v32);
    v32 = (char *)v11 - (char *)v10;
  }
  v33 = (__int64 *)((char *)a6 + v32);
  if ( v9 == v10 )
  {
    if ( a6 == v33 )
      return v7;
LABEL_90:
    v45 = a6;
    v46 = (__m128i *)((char *)v11 - v32);
    goto LABEL_73;
  }
  if ( a6 == v33 )
    return v7;
  v34 = v10 - 4;
  v35 = v33 - 8;
  v11 -= 4;
  while ( 2 )
  {
    v36 = v34[3].m128i_u32[0];
    if ( *((_DWORD *)v35 + 12) == v36 )
    {
      v41 = *((_DWORD *)v35 + 14);
      v42 = v34[3].m128i_i32[2];
      if ( v41 )
      {
        if ( v42 )
        {
          v43 = *v35;
          if ( v41 == 3 )
            v43 = sub_2739680(*v35);
          v44 = v34->m128i_i64[0];
          if ( v42 == 3 )
            v44 = sub_2739680(v34->m128i_i64[0]);
          if ( sub_B445A0(v43, v44) )
            goto LABEL_46;
        }
      }
      else
      {
        if ( v42 )
          goto LABEL_46;
        if ( *(_BYTE *)v35[1] != 17 )
          LOBYTE(v41) = *(_BYTE *)v35[2] != 17;
        if ( *(_BYTE *)v34->m128i_i64[1] != 17
          && (unsigned __int8)v41 < (unsigned __int8)(*(_BYTE *)v34[1].m128i_i64[0] != 17) )
        {
          goto LABEL_46;
        }
      }
LABEL_51:
      *v11 = _mm_loadu_si128((const __m128i *)v35);
      v11[1] = _mm_loadu_si128((const __m128i *)v35 + 1);
      v11[2] = _mm_loadu_si128((const __m128i *)v35 + 2);
      v11[3].m128i_i64[0] = v35[6];
      LODWORD(v7) = *((_DWORD *)v35 + 14);
      v11[3].m128i_i32[2] = v7;
      if ( a6 == v35 )
        return v7;
      v35 -= 8;
LABEL_48:
      v11 -= 4;
      continue;
    }
    break;
  }
  if ( *((_DWORD *)v35 + 12) >= v36 )
    goto LABEL_51;
LABEL_46:
  *v11 = _mm_loadu_si128(v34);
  v11[1] = _mm_loadu_si128(v34 + 1);
  v11[2] = _mm_loadu_si128(v34 + 2);
  v11[3].m128i_i64[0] = v34[3].m128i_i64[0];
  LODWORD(v7) = v34[3].m128i_i32[2];
  v11[3].m128i_i32[2] = v7;
  if ( v34 != v9 )
  {
    v34 -= 4;
    goto LABEL_48;
  }
  if ( a6 != v35 + 8 )
  {
    v32 = (char *)(v35 + 8) - (char *)a6;
    goto LABEL_90;
  }
  return v7;
}

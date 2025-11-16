// Function: sub_12C7BD0
// Address: 0x12c7bd0
//
__int64 __fastcall sub_12C7BD0(__int64 a1, const void *a2, size_t a3, int *a4, char *a5, char a6, char a7)
{
  __int64 v7; // r12
  __int8 *v9; // rdi
  __int64 v11; // rsi
  __int64 v12; // r13
  __int64 v13; // rcx
  char *v14; // r9
  __int64 v15; // rdx
  unsigned __int64 v16; // r15
  int v17; // eax
  __m128i *v18; // r8
  unsigned int v19; // edx
  __m128i *v20; // rax
  __int64 v21; // rdx
  unsigned __int64 v22; // r12
  __int64 v23; // rdi
  __int64 v24; // rdi
  unsigned __int64 v25; // r12
  char *v26; // r14
  size_t v27; // rax
  unsigned int v28; // r14d
  const __m128i *v29; // r12
  __m128i *v30; // rax
  size_t v31; // rax
  unsigned __int64 v32; // r12
  char *v33; // rsi
  const char *v34; // r10
  __int64 v35; // r14
  size_t v36; // rax
  __int64 v37; // r9
  __int8 *v38; // r10
  size_t v39; // r11
  __m128i *v40; // rdx
  unsigned int v41; // edx
  __m128i *v42; // rax
  __int64 v43; // rax
  __m128i *v44; // rdi
  int v45; // eax
  __int64 v46; // rsi
  const __m128i *v47; // rdx
  __m128i *v48; // rcx
  const __m128i *v49; // rsi
  const __m128i *v50; // rbx
  size_t n; // [rsp+10h] [rbp-210h]
  __m128i *v56; // [rsp+48h] [rbp-1D8h]
  char *v57; // [rsp+48h] [rbp-1D8h]
  __m128i *v58; // [rsp+48h] [rbp-1D8h]
  __int8 *v59; // [rsp+48h] [rbp-1D8h]
  __int64 v60; // [rsp+50h] [rbp-1D0h]
  __int64 v62; // [rsp+68h] [rbp-1B8h] BYREF
  __m128i *v63; // [rsp+70h] [rbp-1B0h] BYREF
  size_t v64; // [rsp+78h] [rbp-1A8h]
  __m128i v65; // [rsp+80h] [rbp-1A0h] BYREF
  const __m128i *v66; // [rsp+90h] [rbp-190h] BYREF
  __int64 v67; // [rsp+98h] [rbp-188h]
  _BYTE v68[64]; // [rsp+A0h] [rbp-180h] BYREF
  __m128i *v69; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v70; // [rsp+E8h] [rbp-138h]
  _BYTE dest[304]; // [rsp+F0h] [rbp-130h] BYREF

  v7 = *a4;
  if ( (int)v7 <= 0 )
  {
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0x200000000LL;
    return a1;
  }
  v9 = dest;
  v69 = (__m128i *)dest;
  v70 = 0x2000000000LL;
  if ( (unsigned __int64)(8 * v7) > 0x100 )
  {
    sub_16CD150(&v69, dest, v7, 8);
    v9 = &v69->m128i_i8[8 * (unsigned int)v70];
  }
  v11 = (__int64)a5;
  v12 = 0;
  memcpy(v9, a5, 8 * v7);
  v15 = (unsigned int)v70 + v7;
  v16 = 0;
  v66 = (const __m128i *)v68;
  LODWORD(v70) = v15;
  v67 = 0x200000000LL;
  if ( !(_DWORD)v15 )
  {
    *a4 = 0;
    *(_QWORD *)(a1 + 8) = 0x200000000LL;
    *(_QWORD *)a1 = a1 + 16;
    goto LABEL_34;
  }
  do
  {
    while ( 1 )
    {
      v18 = v69;
      v25 = 0;
      v60 = 8 * v16;
      v26 = (char *)v69->m128i_i64[v16];
      if ( v26 )
      {
        v58 = v69;
        v27 = strlen((const char *)v69->m128i_i64[v16]);
        v18 = v58;
        v25 = v27;
      }
      if ( v25 < a3 )
        break;
      if ( a3 )
      {
        v56 = v18;
        v17 = memcmp(v26, a2, a3);
        v18 = v56;
        if ( v17 )
          break;
      }
      if ( v25 == a3 )
      {
        v22 = v16 + 1;
        v34 = (const char *)v18->m128i_i64[(unsigned __int64)v60 / 8 + 1];
        v35 = v60 + 8;
        v63 = &v65;
        if ( !v34 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v59 = (__int8 *)v34;
        v36 = strlen(v34);
        v38 = v59;
        v18 = &v65;
        v62 = v36;
        v39 = v36;
        if ( v36 > 0xF )
        {
          n = v36;
          v43 = sub_22409D0(&v63, &v62, 0);
          v38 = v59;
          v39 = n;
          v63 = (__m128i *)v43;
          v44 = (__m128i *)v43;
          v65.m128i_i64[0] = v62;
        }
        else
        {
          if ( v36 == 1 )
          {
            v65.m128i_i8[0] = *v59;
            v40 = &v65;
            goto LABEL_52;
          }
          if ( !v36 )
          {
            v40 = &v65;
LABEL_52:
            v64 = v36;
            v40->m128i_i8[v36] = 0;
            v41 = v67;
            if ( (unsigned int)v67 >= HIDWORD(v67) )
            {
              sub_12BE710((__int64)&v66, 0, (unsigned int)v67, v13, (__int64)&v65, v37);
              v41 = v67;
              v18 = &v65;
            }
            v42 = (__m128i *)&v66[2 * v41];
            if ( v42 )
            {
              v42->m128i_i64[0] = (__int64)v42[1].m128i_i64;
              if ( v63 == &v65 )
              {
                v42[1] = _mm_load_si128(&v65);
              }
              else
              {
                v42->m128i_i64[0] = (__int64)v63;
                v42[1].m128i_i64[0] = v65.m128i_i64[0];
              }
              v15 = v64;
              v42->m128i_i64[1] = v64;
              LODWORD(v67) = v67 + 1;
            }
            else
            {
              v15 = v41 + 1;
              LODWORD(v67) = v15;
              if ( v63 != &v65 )
                j_j___libc_free_0(v63, v65.m128i_i64[0] + 1);
            }
            v14 = &a5[v35];
            v23 = *(_QWORD *)&a5[v35];
            if ( v23 )
            {
LABEL_21:
              v57 = v14;
              j_j___libc_free_0_0(v23);
              v14 = v57;
            }
            *(_QWORD *)v14 = 0;
            if ( v16 != v22 )
            {
              v24 = *(_QWORD *)&a5[8 * v16];
              if ( v24 )
                j_j___libc_free_0_0(v24);
              *(_QWORD *)&a5[8 * v16] = 0;
              v16 = v22;
            }
            goto LABEL_26;
          }
          v44 = &v65;
        }
        memcpy(v44, v38, v39);
        v36 = v62;
        v40 = v63;
        v18 = &v65;
        goto LABEL_52;
      }
      if ( a6 )
      {
        if ( &v26[a3] )
        {
          v63 = &v65;
          sub_12C6150((__int64 *)&v63, &v26[a3], (__int64)&v26[v25]);
          v18 = &v65;
        }
        else
        {
          v18 = &v65;
          v65.m128i_i8[0] = 0;
          v63 = &v65;
          v64 = 0;
        }
        v19 = v67;
        if ( (unsigned int)v67 >= HIDWORD(v67) )
        {
          sub_12BE710((__int64)&v66, 0, (unsigned int)v67, v13, (__int64)&v65, (__int64)v14);
          v19 = v67;
          v18 = &v65;
        }
        v20 = (__m128i *)&v66[2 * v19];
        if ( v20 )
        {
          v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
          v21 = (__int64)v63;
          if ( v63 != &v65 )
            goto LABEL_18;
LABEL_45:
          v20[1] = _mm_load_si128(&v65);
          goto LABEL_19;
        }
        goto LABEL_46;
      }
      if ( !a7 || a7 != v26[a3] )
        break;
      v31 = a3 + 1;
      if ( v25 >= a3 + 1 )
      {
        v32 = v25 - v31;
        v33 = &v26[v31];
LABEL_40:
        v63 = &v65;
        sub_12C6150((__int64 *)&v63, v33, (__int64)&v33[v32]);
        v18 = &v65;
        goto LABEL_41;
      }
      v33 = &v26[v25];
      if ( &v26[v25] )
      {
        v32 = 0;
        goto LABEL_40;
      }
      v18 = &v65;
      v65.m128i_i8[0] = 0;
      v63 = &v65;
      v64 = 0;
LABEL_41:
      v19 = v67;
      if ( (unsigned int)v67 >= HIDWORD(v67) )
      {
        sub_12BE710((__int64)&v66, 0, (unsigned int)v67, v13, (__int64)&v65, (__int64)v14);
        v19 = v67;
        v18 = &v65;
      }
      v20 = (__m128i *)&v66[2 * v19];
      if ( v20 )
      {
        v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
        v21 = (__int64)v63;
        if ( v63 == &v65 )
          goto LABEL_45;
LABEL_18:
        v20->m128i_i64[0] = v21;
        v20[1].m128i_i64[0] = v65.m128i_i64[0];
LABEL_19:
        v15 = v64;
        v20->m128i_i64[1] = v64;
        LODWORD(v67) = v67 + 1;
        goto LABEL_20;
      }
LABEL_46:
      v15 = v19 + 1;
      LODWORD(v67) = v15;
      if ( v63 != &v65 )
        j_j___libc_free_0(v63, v65.m128i_i64[0] + 1);
LABEL_20:
      v14 = &a5[v60];
      v22 = v16;
      v23 = *(_QWORD *)&a5[v60];
      if ( v23 )
        goto LABEL_21;
LABEL_26:
      if ( ++v16 >= (unsigned int)v70 )
        goto LABEL_31;
    }
    ++v16;
    *(_QWORD *)&a5[8 * v12++] = v26;
  }
  while ( v16 < (unsigned int)v70 );
LABEL_31:
  v11 = a1;
  v28 = v67;
  v29 = v66;
  *a4 = v12;
  v30 = (__m128i *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  if ( !v28 )
    goto LABEL_32;
  if ( v29 == (const __m128i *)v68 )
  {
    v46 = v28;
    if ( v28 > 2 )
    {
      sub_12BE710(a1, v28, v15, v13, (__int64)v18, (__int64)v14);
      v30 = *(__m128i **)a1;
      v29 = v66;
      v46 = (unsigned int)v67;
    }
    v11 = 32 * v46;
    if ( v11 )
    {
      v47 = v29 + 1;
      v48 = (__m128i *)((char *)v30 + v11);
      do
      {
        if ( v30 )
        {
          v30->m128i_i64[0] = (__int64)v30[1].m128i_i64;
          v49 = (const __m128i *)v47[-1].m128i_i64[0];
          if ( v49 == v47 )
          {
            v30[1] = _mm_loadu_si128(v47);
          }
          else
          {
            v30->m128i_i64[0] = (__int64)v49;
            v30[1].m128i_i64[0] = v47->m128i_i64[0];
          }
          v11 = v47[-1].m128i_i64[1];
          v30->m128i_i64[1] = v11;
          v47[-1].m128i_i64[0] = (__int64)v47;
          v47[-1].m128i_i64[1] = 0;
          v47->m128i_i8[0] = 0;
        }
        v30 += 2;
        v47 += 2;
      }
      while ( v30 != v48 );
      v29 = v66;
      v50 = &v66[2 * (unsigned int)v67];
      *(_DWORD *)(a1 + 8) = v28;
      if ( v50 != v29 )
      {
        do
        {
          v50 -= 2;
          if ( (const __m128i *)v50->m128i_i64[0] != &v50[1] )
          {
            v11 = v50[1].m128i_i64[0] + 1;
            j_j___libc_free_0(v50->m128i_i64[0], v11);
          }
        }
        while ( v29 != v50 );
        v29 = v66;
      }
    }
    else
    {
      *(_DWORD *)(a1 + 8) = v28;
    }
LABEL_32:
    if ( v29 != (const __m128i *)v68 )
      _libc_free(v29, v11);
  }
  else
  {
    v45 = HIDWORD(v67);
    *(_QWORD *)a1 = v29;
    *(_DWORD *)(a1 + 8) = v28;
    *(_DWORD *)(a1 + 12) = v45;
  }
LABEL_34:
  if ( v69 != (__m128i *)dest )
    _libc_free(v69, v11);
  return a1;
}

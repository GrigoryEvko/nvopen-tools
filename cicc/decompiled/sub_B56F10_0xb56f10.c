// Function: sub_B56F10
// Address: 0xb56f10
//
__int64 __fastcall sub_B56F10(__int64 a1, __m128i *a2, __int64 a3, unsigned __int16 a4)
{
  __m128i *v5; // rbx
  bool v6; // sf
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // r12
  int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rcx
  size_t *v18; // rdi
  __int64 v19; // rax
  size_t v20; // rdx
  __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  __m128i *v23; // r12
  __int64 *v24; // rcx
  __m128i *v25; // r8
  size_t v26; // r12
  _BYTE *v27; // rdi
  unsigned __int64 v28; // r12
  __int64 v29; // rax
  char *v30; // rdi
  _BYTE *v31; // rax
  _BYTE *v32; // rsi
  size_t v33; // rbx
  char *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // rax
  __m128i *v38; // rbx
  __int64 v39; // r13
  __m128i *v40; // r12
  __int64 v41; // rdi
  __m128i *v43; // rax
  __int64 v44; // rdi
  int v45; // edx
  __m128i *v46; // rax
  __int64 v47; // rax
  __m128i *v48; // r10
  size_t v49; // r8
  __int64 v50; // rax
  unsigned __int64 v51; // rdx
  char *v52; // rdi
  _BYTE *v53; // rax
  _BYTE *v54; // rsi
  size_t v55; // rbx
  char *v56; // rax
  int v57; // ebx
  __int64 v58; // rax
  size_t n; // [rsp+8h] [rbp-108h]
  int src; // [rsp+10h] [rbp-100h]
  __m128i *srca; // [rsp+10h] [rbp-100h]
  __int64 *srcb; // [rsp+10h] [rbp-100h]
  __m128i *srcc; // [rsp+10h] [rbp-100h]
  __int64 *v64; // [rsp+18h] [rbp-F8h]
  __int64 *v65; // [rsp+18h] [rbp-F8h]
  __m128i *v66; // [rsp+18h] [rbp-F8h]
  __m128i *v67; // [rsp+18h] [rbp-F8h]
  __int64 *v68; // [rsp+18h] [rbp-F8h]
  __int64 *v69; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v70; // [rsp+18h] [rbp-F8h]
  __int64 *v71; // [rsp+18h] [rbp-F8h]
  __int64 *v72; // [rsp+18h] [rbp-F8h]
  __int64 *v73; // [rsp+18h] [rbp-F8h]
  __int64 v76; // [rsp+38h] [rbp-D8h] BYREF
  _QWORD v77[4]; // [rsp+40h] [rbp-D0h] BYREF
  __m128i *v78; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v79; // [rsp+68h] [rbp-A8h]
  _BYTE v80[160]; // [rsp+70h] [rbp-A0h] BYREF

  v5 = a2;
  v6 = *(char *)(a1 + 7) < 0;
  v78 = (__m128i *)v80;
  v79 = 0x200000000LL;
  if ( !v6 )
  {
    v21 = (unsigned int)v79;
    v22 = (unsigned int)v79;
LABEL_18:
    v23 = v78;
    v24 = &v78->m128i_i64[7 * v21];
    if ( !v24 )
    {
LABEL_30:
      v35 = (unsigned int)(v22 + 1);
      LODWORD(v79) = v35;
      goto LABEL_31;
    }
    *v24 = (__int64)(v24 + 2);
    v25 = (__m128i *)v5->m128i_i64[0];
    v26 = v5->m128i_u64[1];
    if ( !(v26 + v5->m128i_i64[0]) || v25 )
    {
      v77[0] = v5->m128i_i64[1];
      if ( v26 > 0xF )
      {
        srca = v25;
        v68 = v24;
        v47 = sub_22409D0(v24, v77, 0);
        v24 = v68;
        v25 = srca;
        v27 = (_BYTE *)v47;
        *v68 = v47;
        v68[2] = v77[0];
      }
      else
      {
        v27 = (_BYTE *)*v24;
        if ( v26 == 1 )
        {
          *v27 = v25->m128i_i8[0];
          v26 = v77[0];
          v27 = (_BYTE *)*v24;
LABEL_24:
          v24[1] = v26;
          v27[v26] = 0;
          v28 = v5[2].m128i_i64[1] - v5[2].m128i_i64[0];
          v24[4] = 0;
          v24[5] = 0;
          v24[6] = 0;
          if ( !v28 )
          {
            v30 = 0;
            goto LABEL_27;
          }
          v64 = v24;
          if ( v28 <= 0x7FFFFFFFFFFFFFF8LL )
          {
            v29 = sub_22077B0(v28);
            v24 = v64;
            v30 = (char *)v29;
LABEL_27:
            v24[4] = (__int64)v30;
            v24[5] = (__int64)v30;
            v24[6] = (__int64)&v30[v28];
            v31 = (_BYTE *)v5[2].m128i_i64[1];
            v32 = (_BYTE *)v5[2].m128i_i64[0];
            v33 = v31 - v32;
            if ( v31 != v32 )
            {
              v65 = v24;
              v34 = (char *)memmove(v30, v32, v33);
              v24 = v65;
              v30 = v34;
            }
            LODWORD(v22) = v79;
            v23 = v78;
            v24[5] = (__int64)&v30[v33];
            goto LABEL_30;
          }
LABEL_74:
          sub_4261EA(v27, a2, v22, v24);
        }
        if ( !v26 )
          goto LABEL_24;
      }
      a2 = v25;
      v69 = v24;
      memcpy(v27, v25, v26);
      v24 = v69;
      v26 = v77[0];
      v27 = (_BYTE *)*v69;
      goto LABEL_24;
    }
LABEL_55:
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  }
  v7 = sub_BD2BC0(a1);
  v9 = v7 + v8;
  if ( *(char *)(a1 + 7) >= 0 )
    v10 = v9 >> 4;
  else
    LODWORD(v10) = (v9 - sub_BD2BC0(a1)) >> 4;
  v11 = 0;
  v12 = 16LL * (unsigned int)v10;
  if ( (_DWORD)v10 )
  {
    while ( 1 )
    {
      v15 = 0;
      if ( *(char *)(a1 + 7) < 0 )
        v15 = sub_BD2BC0(a1);
      v16 = v11 + v15;
      v17 = *(unsigned int *)(v16 + 8);
      v18 = *(size_t **)v16;
      v19 = *(unsigned int *)(v16 + 12);
      v17 *= 32;
      a2 = (__m128i *)(32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      v77[2] = v18;
      v77[0] = a1 + v17 - (_QWORD)a2;
      v77[1] = (32 * v19 - v17) >> 5;
      v20 = *v18;
      if ( *v18 != v5->m128i_i64[1] )
        break;
      if ( v20 )
      {
        a2 = (__m128i *)v5->m128i_i64[0];
        if ( memcmp(v18 + 2, (const void *)v5->m128i_i64[0], v20) )
          break;
        v11 += 16;
        if ( v12 == v11 )
          goto LABEL_17;
      }
      else
      {
LABEL_10:
        v11 += 16;
        if ( v12 == v11 )
          goto LABEL_17;
      }
    }
    v13 = v79;
    if ( HIDWORD(v79) <= (unsigned int)v79 )
    {
      v43 = (__m128i *)sub_C8D7D0(&v78, v80, 0, 56, &v76);
      v44 = (__int64)&v43->m128i_i64[7 * (unsigned int)v79];
      if ( v44 )
      {
        v66 = v43;
        sub_B56460(v44, (__int64)v77);
        v43 = v66;
      }
      a2 = v43;
      v67 = v43;
      sub_B56820((__int64)&v78, v43);
      v45 = v76;
      v46 = v67;
      if ( v78 != (__m128i *)v80 )
      {
        src = v76;
        _libc_free(v78, a2);
        v45 = src;
        v46 = v67;
      }
      LODWORD(v79) = v79 + 1;
      v78 = v46;
      HIDWORD(v79) = v45;
    }
    else
    {
      v14 = (__int64)&v78->m128i_i64[7 * (unsigned int)v79];
      if ( v14 )
      {
        a2 = (__m128i *)v77;
        sub_B56460(v14, (__int64)v77);
        v13 = v79;
      }
      LODWORD(v79) = v13 + 1;
    }
    goto LABEL_10;
  }
LABEL_17:
  v21 = (unsigned int)v79;
  v22 = (unsigned int)v79;
  if ( HIDWORD(v79) > (unsigned int)v79 )
    goto LABEL_18;
  a2 = (__m128i *)v80;
  v23 = (__m128i *)sub_C8D7D0(&v78, v80, 0, 56, &v76);
  v24 = &v23->m128i_i64[7 * (unsigned int)v79];
  if ( v24 )
  {
    *v24 = (__int64)(v24 + 2);
    v48 = (__m128i *)v5->m128i_i64[0];
    v49 = v5->m128i_u64[1];
    if ( v49 + v5->m128i_i64[0] && !v48 )
      goto LABEL_55;
    v77[0] = v5->m128i_i64[1];
    if ( v49 > 0xF )
    {
      n = v49;
      srcc = v48;
      v72 = v24;
      v58 = sub_22409D0(v24, v77, 0);
      v24 = v72;
      v48 = srcc;
      v27 = (_BYTE *)v58;
      v49 = n;
      *v72 = v58;
      v72[2] = v77[0];
    }
    else
    {
      v27 = (_BYTE *)*v24;
      if ( v49 == 1 )
      {
        *v27 = v48->m128i_i8[0];
        v49 = v77[0];
        v27 = (_BYTE *)*v24;
LABEL_59:
        v24[1] = v49;
        v27[v49] = 0;
        v22 = v5[2].m128i_i64[1] - v5[2].m128i_i64[0];
        v24[4] = 0;
        v24[5] = 0;
        v24[6] = 0;
        if ( v22 )
        {
          if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_74;
          srcb = v24;
          v70 = v22;
          v50 = sub_22077B0(v22);
          v51 = v70;
          v24 = srcb;
          v52 = (char *)v50;
        }
        else
        {
          v51 = 0;
          v52 = 0;
        }
        v24[4] = (__int64)v52;
        v24[5] = (__int64)v52;
        v24[6] = (__int64)&v52[v51];
        v53 = (_BYTE *)v5[2].m128i_i64[1];
        v54 = (_BYTE *)v5[2].m128i_i64[0];
        v55 = v53 - v54;
        if ( v53 != v54 )
        {
          v71 = v24;
          v56 = (char *)memmove(v52, v54, v55);
          v24 = v71;
          v52 = v56;
        }
        v24[5] = (__int64)&v52[v55];
        goto LABEL_65;
      }
      if ( !v49 )
        goto LABEL_59;
    }
    a2 = v48;
    v73 = v24;
    memcpy(v27, v48, v49);
    v24 = v73;
    v49 = v77[0];
    v27 = (_BYTE *)*v73;
    goto LABEL_59;
  }
LABEL_65:
  sub_B56820((__int64)&v78, v23);
  v57 = v76;
  if ( v78 != (__m128i *)v80 )
    _libc_free(v78, v23);
  v78 = v23;
  HIDWORD(v79) = v57;
  v35 = (unsigned int)(v79 + 1);
  LODWORD(v79) = v79 + 1;
LABEL_31:
  v36 = (__int64)v23;
  v37 = sub_B4BA60((unsigned __int8 *)a1, (__int64)v23, v35, a3, a4);
  v38 = v78;
  v39 = v37;
  v40 = (__m128i *)((char *)v78 + 56 * (unsigned int)v79);
  if ( v78 != v40 )
  {
    do
    {
      v41 = v40[-2].m128i_i64[1];
      v40 = (__m128i *)((char *)v40 - 56);
      if ( v41 )
      {
        v36 = v40[3].m128i_i64[0] - v41;
        j_j___libc_free_0(v41, v36);
      }
      if ( (__m128i *)v40->m128i_i64[0] != &v40[1] )
      {
        v36 = v40[1].m128i_i64[0] + 1;
        j_j___libc_free_0(v40->m128i_i64[0], v36);
      }
    }
    while ( v38 != v40 );
    v40 = v78;
  }
  if ( v40 != (__m128i *)v80 )
    _libc_free(v40, v36);
  return v39;
}

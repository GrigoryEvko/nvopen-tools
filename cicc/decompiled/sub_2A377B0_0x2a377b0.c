// Function: sub_2A377B0
// Address: 0x2a377b0
//
void __fastcall sub_2A377B0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v3; // r15
  __int64 v5; // r9
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  const __m128i *v10; // r12
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r8
  __m128i *v13; // rax
  unsigned __int64 *v14; // r14
  unsigned __int64 v15; // rax
  char v16; // r12
  unsigned __int64 *v17; // rax
  unsigned __int64 v18; // r13
  unsigned __int64 *v19; // r14
  unsigned __int64 v20; // rax
  unsigned __int64 *v21; // rax
  unsigned __int64 v22; // r13
  __int64 v23; // r9
  __int64 v24; // rdx
  const __m128i *v25; // rax
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // r8
  __m128i *v28; // rdx
  unsigned __int64 *v29; // rcx
  unsigned __int64 *v30; // r13
  unsigned __int64 *v31; // r15
  __int64 v32; // rdi
  __int64 v33; // rdx
  __int64 v34; // r8
  unsigned __int64 v35; // rcx
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  const __m128i *v38; // r12
  unsigned __int64 v39; // r9
  __m128i *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // r14
  char v44; // r12
  unsigned __int64 v45; // rax
  char v46; // dl
  unsigned __int8 v47; // al
  unsigned __int64 *v48; // rcx
  unsigned __int64 *v49; // r15
  char v50; // si
  unsigned __int64 *v51; // r14
  __int64 v52; // rdi
  __int64 v53; // rdx
  __int64 v54; // r8
  __int64 v55; // r9
  unsigned __int64 v56; // rcx
  const __m128i *v57; // r13
  unsigned __int64 v58; // rdx
  __int64 v59; // rax
  unsigned __int64 v60; // r10
  __m128i *v61; // rax
  __int64 v62; // rax
  __int64 v63; // r13
  char v64; // r12
  unsigned __int64 v65; // rax
  char v66; // dl
  unsigned __int8 v67; // al
  __int64 v68; // rdx
  char *v69; // r12
  const void *v70; // rsi
  char *v71; // r13
  const void *v72; // rsi
  char *v73; // r12
  const char *v74; // rax
  __int64 v75; // rdx
  const void *v76; // rsi
  char *v77; // r12
  unsigned __int64 v78; // [rsp+8h] [rbp-D8h]
  const void *v79; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v82; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v83; // [rsp+30h] [rbp-B0h]
  _BYTE *v84; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v85; // [rsp+60h] [rbp-80h] BYREF
  _BYTE v86[24]; // [rsp+68h] [rbp-78h] BYREF
  __int64 v87; // [rsp+80h] [rbp-60h] BYREF
  __int64 v88; // [rsp+88h] [rbp-58h]
  char v89; // [rsp+90h] [rbp-50h]
  unsigned __int64 v90; // [rsp+98h] [rbp-48h]
  char v91; // [rsp+A0h] [rbp-40h]

  v3 = (_BYTE *)a2;
  if ( *(_BYTE *)a2 == 3 )
  {
    v87 = sub_9208B0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a2 + 24));
    v6 = v87;
    v88 = v7;
    if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
    {
      v89 = 1;
      v87 = (__int64)sub_BD5D20(a2);
      v88 = v68;
    }
    else
    {
      v89 = 0;
    }
    v8 = *(unsigned int *)(a3 + 8);
    v9 = *(unsigned int *)(a3 + 12);
    v90 = v6;
    v10 = (const __m128i *)&v87;
    v91 = 1;
    v11 = *(_QWORD *)a3;
    v12 = v8 + 1;
    if ( v8 + 1 > v9 )
    {
      v72 = (const void *)(a3 + 16);
      if ( v11 > (unsigned __int64)&v87 || (unsigned __int64)&v87 >= v11 + 40 * v8 )
      {
        sub_C8D5F0(a3, v72, v12, 0x28u, v12, v5);
        v11 = *(_QWORD *)a3;
        v8 = *(unsigned int *)(a3 + 8);
      }
      else
      {
        v73 = (char *)&v87 - v11;
        sub_C8D5F0(a3, v72, v12, 0x28u, v12, v5);
        v11 = *(_QWORD *)a3;
        v8 = *(unsigned int *)(a3 + 8);
        v10 = (const __m128i *)&v73[*(_QWORD *)a3];
      }
    }
    v13 = (__m128i *)(v11 + 40 * v8);
    *v13 = _mm_loadu_si128(v10);
    v13[1] = _mm_loadu_si128(v10 + 1);
    v13[2].m128i_i64[0] = v10[2].m128i_i64[0];
    ++*(_DWORD *)(a3 + 8);
    return;
  }
  v14 = &v85;
  sub_AE74C0(&v85, a2);
  v15 = v85;
  if ( (v85 & 4) != 0 )
  {
    v14 = *(unsigned __int64 **)(v85 & 0xFFFFFFFFFFFFFFF8LL);
    v48 = &v14[*(unsigned int *)((v85 & 0xFFFFFFFFFFFFFFF8LL) + 8)];
    if ( v14 == v48 )
    {
      v16 = 0;
      goto LABEL_9;
    }
    goto LABEL_47;
  }
  v16 = 0;
  if ( (v85 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v48 = (unsigned __int64 *)v86;
LABEL_47:
    v49 = v14;
    v50 = 0;
    v51 = v48;
    while ( 1 )
    {
      v62 = *(_QWORD *)(*v49 + 32 * (1LL - (*(_DWORD *)(*v49 + 4) & 0x7FFFFFF)));
      v63 = *(_QWORD *)(v62 + 24);
      if ( v63 )
        break;
LABEL_52:
      if ( ++v49 == v51 )
      {
        v3 = (_BYTE *)a2;
        v15 = v85;
        v16 = v50;
        goto LABEL_9;
      }
    }
    v64 = 0;
    v65 = sub_AF3FE0(*(_QWORD *)(v62 + 24));
    if ( v66 && (v65 & 7) == 0 )
    {
      v64 = 1;
      v83 = v65 >> 3;
    }
    v67 = *(_BYTE *)(v63 - 16);
    if ( (v67 & 2) != 0 )
    {
      v52 = *(_QWORD *)(*(_QWORD *)(v63 - 32) + 8LL);
      if ( !v52 )
      {
LABEL_59:
        v53 = 0;
        goto LABEL_50;
      }
    }
    else
    {
      v52 = *(_QWORD *)(v63 - 16 - 8LL * ((v67 >> 2) & 0xF) + 8);
      if ( !v52 )
        goto LABEL_59;
    }
    v52 = sub_B91420(v52);
LABEL_50:
    v56 = *(unsigned int *)(a3 + 12);
    v88 = v53;
    v57 = (const __m128i *)&v87;
    v58 = *(_QWORD *)a3;
    v87 = v52;
    v90 = v83;
    v59 = *(unsigned int *)(a3 + 8);
    v89 = 1;
    v60 = v59 + 1;
    v91 = v64;
    if ( v59 + 1 > v56 )
    {
      v70 = (const void *)(a3 + 16);
      if ( v58 > (unsigned __int64)&v87 || (unsigned __int64)&v87 >= v58 + 40 * v59 )
      {
        sub_C8D5F0(a3, v70, v60, 0x28u, v54, v55);
        v58 = *(_QWORD *)a3;
        v59 = *(unsigned int *)(a3 + 8);
        v57 = (const __m128i *)&v87;
      }
      else
      {
        v71 = (char *)&v87 - v58;
        sub_C8D5F0(a3, v70, v60, 0x28u, v54, v55);
        v58 = *(_QWORD *)a3;
        v59 = *(unsigned int *)(a3 + 8);
        v57 = (const __m128i *)&v71[*(_QWORD *)a3];
      }
    }
    v50 = 1;
    v61 = (__m128i *)(v58 + 40 * v59);
    *v61 = _mm_loadu_si128(v57);
    v61[1] = _mm_loadu_si128(v57 + 1);
    v61[2].m128i_i64[0] = v57[2].m128i_i64[0];
    ++*(_DWORD *)(a3 + 8);
    goto LABEL_52;
  }
LABEL_9:
  if ( v15 )
  {
    if ( (v15 & 4) != 0 )
    {
      v17 = (unsigned __int64 *)(v15 & 0xFFFFFFFFFFFFFFF8LL);
      v18 = (unsigned __int64)v17;
      if ( v17 )
      {
        if ( (unsigned __int64 *)*v17 != v17 + 2 )
          _libc_free(*v17);
        j_j___libc_free_0(v18);
      }
    }
  }
  v19 = &v85;
  sub_AE7690(&v85, (__int64)v3);
  v20 = v85;
  if ( (v85 & 4) != 0 )
  {
    v19 = *(unsigned __int64 **)(v85 & 0xFFFFFFFFFFFFFFF8LL);
    v29 = &v19[*(unsigned int *)((v85 & 0xFFFFFFFFFFFFFFF8LL) + 8)];
    if ( v19 == v29 )
      goto LABEL_17;
    goto LABEL_32;
  }
  if ( (v85 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v29 = (unsigned __int64 *)v86;
LABEL_32:
    v84 = v3;
    v30 = v29;
    v31 = v19;
    v79 = (const void *)(a3 + 16);
    while ( 1 )
    {
      v42 = sub_B12000(*v31 + 72);
      v43 = v42;
      if ( v42 )
        break;
LABEL_37:
      if ( ++v31 == v30 )
      {
        v3 = v84;
        v20 = v85;
        goto LABEL_17;
      }
    }
    v44 = 0;
    v45 = sub_AF3FE0(v42);
    if ( v46 && (v45 & 7) == 0 )
    {
      v44 = 1;
      v82 = v45 >> 3;
    }
    v47 = *(_BYTE *)(v43 - 16);
    if ( (v47 & 2) != 0 )
    {
      v32 = *(_QWORD *)(*(_QWORD *)(v43 - 32) + 8LL);
      if ( !v32 )
      {
LABEL_44:
        v33 = 0;
        goto LABEL_35;
      }
    }
    else
    {
      v32 = *(_QWORD *)(v43 - 16 - 8LL * ((v47 >> 2) & 0xF) + 8);
      if ( !v32 )
        goto LABEL_44;
    }
    v32 = sub_B91420(v32);
LABEL_35:
    v35 = *(unsigned int *)(a3 + 12);
    v88 = v33;
    v91 = v44;
    v36 = *(_QWORD *)a3;
    v90 = v82;
    v37 = *(unsigned int *)(a3 + 8);
    v87 = v32;
    v38 = (const __m128i *)&v87;
    v39 = v37 + 1;
    v89 = 1;
    if ( v37 + 1 > v35 )
    {
      if ( v36 > (unsigned __int64)&v87 || (unsigned __int64)&v87 >= v36 + 40 * v37 )
      {
        sub_C8D5F0(a3, v79, v39, 0x28u, v34, v39);
        v36 = *(_QWORD *)a3;
        v37 = *(unsigned int *)(a3 + 8);
        v38 = (const __m128i *)&v87;
      }
      else
      {
        v69 = (char *)&v87 - v36;
        sub_C8D5F0(a3, v79, v39, 0x28u, v34, v39);
        v36 = *(_QWORD *)a3;
        v37 = *(unsigned int *)(a3 + 8);
        v38 = (const __m128i *)&v69[*(_QWORD *)a3];
      }
    }
    v40 = (__m128i *)(v36 + 40 * v37);
    *v40 = _mm_loadu_si128(v38);
    v40[1] = _mm_loadu_si128(v38 + 1);
    v41 = v38[2].m128i_i64[0];
    v16 = 1;
    v40[2].m128i_i64[0] = v41;
    ++*(_DWORD *)(a3 + 8);
    goto LABEL_37;
  }
LABEL_17:
  if ( v20 )
  {
    if ( (v20 & 4) != 0 )
    {
      v21 = (unsigned __int64 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
      v22 = (unsigned __int64)v21;
      if ( v21 )
      {
        if ( (unsigned __int64 *)*v21 != v21 + 2 )
          _libc_free(*v21);
        j_j___libc_free_0(v22);
      }
    }
  }
  if ( !v16 && *v3 == 60 )
  {
    sub_B4CED0((__int64)&v85, (__int64)v3, *(_QWORD *)(a1 + 32));
    if ( v86[8] )
    {
      v16 = 1;
      v78 = v85;
    }
    if ( (v3[7] & 0x10) != 0 )
    {
      v74 = sub_BD5D20((__int64)v3);
      v89 = 1;
      v87 = (__int64)v74;
      v88 = v75;
      v90 = v78;
      v91 = v16;
    }
    else
    {
      v89 = 0;
      v91 = v16;
      v90 = v78;
      if ( !v16 )
        return;
    }
    v24 = *(unsigned int *)(a3 + 8);
    v25 = (const __m128i *)&v87;
    v26 = *(_QWORD *)a3;
    v27 = v24 + 1;
    if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      v76 = (const void *)(a3 + 16);
      if ( v26 > (unsigned __int64)&v87 || (unsigned __int64)&v87 >= v26 + 40 * v24 )
      {
        sub_C8D5F0(a3, v76, v27, 0x28u, v27, v23);
        v26 = *(_QWORD *)a3;
        v24 = *(unsigned int *)(a3 + 8);
        v25 = (const __m128i *)&v87;
      }
      else
      {
        v77 = (char *)&v87 - v26;
        sub_C8D5F0(a3, v76, v27, 0x28u, v27, v23);
        v26 = *(_QWORD *)a3;
        v24 = *(unsigned int *)(a3 + 8);
        v25 = (const __m128i *)&v77[*(_QWORD *)a3];
      }
    }
    v28 = (__m128i *)(v26 + 40 * v24);
    *v28 = _mm_loadu_si128(v25);
    v28[1] = _mm_loadu_si128(v25 + 1);
    v28[2].m128i_i64[0] = v25[2].m128i_i64[0];
    ++*(_DWORD *)(a3 + 8);
  }
}

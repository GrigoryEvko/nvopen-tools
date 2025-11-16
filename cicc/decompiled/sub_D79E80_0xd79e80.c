// Function: sub_D79E80
// Address: 0xd79e80
//
__int64 __fastcall sub_D79E80(_DWORD *a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r12d
  unsigned int v5; // r13d
  __int64 v7; // rax
  __int64 v8; // rbx
  _BYTE *v10; // rax
  _BYTE *v11; // rsi
  size_t v12; // rdx
  signed __int64 v13; // r13
  _DWORD *v14; // rax
  const void *v15; // rsi
  __int64 *v16; // rbx
  __int64 v17; // r13
  _BYTE *v18; // rax
  size_t v19; // r15
  __int64 v20; // rax
  _DWORD *v21; // r12
  _BYTE *v22; // rax
  __int64 v23; // r10
  __int64 v24; // rcx
  unsigned __int64 v25; // r8
  __int64 v26; // rax
  void *v27; // rdi
  size_t v28; // rdx
  int v29; // r13d
  void *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdx
  void *v33; // r13
  bool v34; // r12
  size_t v35; // rcx
  size_t v36; // rsi
  int v37; // eax
  int v38; // [rsp+4h] [rbp-10Ch]
  __int64 v39; // [rsp+8h] [rbp-108h]
  __int64 v40; // [rsp+10h] [rbp-100h]
  unsigned __int64 v41; // [rsp+10h] [rbp-100h]
  __int64 *v42; // [rsp+18h] [rbp-F8h]
  signed __int64 v43; // [rsp+20h] [rbp-F0h]
  signed __int64 v44; // [rsp+28h] [rbp-E8h]
  _QWORD *v45; // [rsp+30h] [rbp-E0h]
  __int64 v46; // [rsp+38h] [rbp-D8h]
  size_t n; // [rsp+40h] [rbp-D0h]
  size_t nb; // [rsp+40h] [rbp-D0h]
  size_t na; // [rsp+40h] [rbp-D0h]
  int v50; // [rsp+48h] [rbp-C8h]
  unsigned int v51; // [rsp+4Ch] [rbp-C4h]
  __m128i v52; // [rsp+50h] [rbp-C0h] BYREF
  const void *v53[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v54; // [rsp+70h] [rbp-A0h]
  __m128i v55; // [rsp+80h] [rbp-90h]
  void *s1; // [rsp+90h] [rbp-80h] BYREF
  __int64 v57; // [rsp+98h] [rbp-78h]
  size_t v58; // [rsp+A0h] [rbp-70h]
  _BYTE v59[24]; // [rsp+B0h] [rbp-60h] BYREF
  unsigned __int64 v60; // [rsp+C8h] [rbp-48h]
  __int64 v61; // [rsp+D0h] [rbp-40h]

  v4 = a1[6];
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v7 = *((_QWORD *)a1 + 1);
  v8 = *a2;
  v52.m128i_i64[0] = 0;
  v52.m128i_i64[1] = -2;
  v46 = v7;
  v10 = (_BYTE *)a2[3];
  v53[0] = 0;
  v11 = (_BYTE *)a2[2];
  v53[1] = 0;
  v54 = 0;
  v12 = v10 - v11;
  v13 = v10 - v11;
  if ( v10 == v11 )
  {
    a1 = 0;
  }
  else
  {
    if ( v12 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_63:
      sub_4261EA(a1, v11, v12);
    v14 = (_DWORD *)sub_22077B0(v12);
    v15 = (const void *)a2[2];
    a1 = v14;
    v12 = a2[3] - (_QWORD)v15;
    if ( (const void *)a2[3] == v15 )
    {
      if ( !v14 )
        goto LABEL_9;
    }
    else
    {
      a1 = memmove(v14, v15, v12);
    }
    j_j___libc_free_0(a1, v13);
  }
LABEL_9:
  v45 = a3;
  v38 = v4 - 1;
  v51 = (v4 - 1) & v8;
  v50 = 1;
  v42 = 0;
  while ( 1 )
  {
    v16 = (__int64 *)(v46 + 40LL * v51);
    v11 = (_BYTE *)v16[2];
    v17 = *v16;
    n = v16[1];
    v18 = (_BYTE *)v16[3];
    v19 = v18 - v11;
    v43 = v18 - v11;
    if ( v18 == v11 )
    {
      v21 = 0;
    }
    else
    {
      if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_63;
      a1 = (_DWORD *)(v16[3] - (_QWORD)v11);
      v20 = sub_22077B0(v19);
      v11 = (_BYTE *)v16[2];
      v21 = (_DWORD *)v20;
      v18 = (_BYTE *)v16[3];
      v19 = v18 - v11;
    }
    if ( v11 != v18 )
    {
      a1 = v21;
      memmove(v21, v11, v19);
    }
    v22 = (_BYTE *)a2[3];
    v11 = (_BYTE *)a2[2];
    v23 = *a2;
    v24 = a2[1];
    v25 = v22 - v11;
    v44 = v22 - v11;
    if ( v22 == v11 )
    {
      v28 = 0;
      v27 = 0;
    }
    else
    {
      v39 = a2[1];
      v40 = *a2;
      if ( v25 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_63;
      v26 = sub_22077B0(v25);
      v11 = (_BYTE *)a2[2];
      v23 = v40;
      v27 = (void *)v26;
      v22 = (_BYTE *)a2[3];
      v24 = v39;
      v25 = v22 - v11;
      v28 = v22 - v11;
    }
    LOBYTE(v17) = v17 == v23;
    LOBYTE(v24) = n == v24;
    v29 = v24 & v17;
    LOBYTE(v24) = v19 == v25;
    v5 = v24 & v29;
    if ( v22 != v11 )
    {
      v41 = v25;
      nb = v28;
      v30 = memmove(v27, v11, v28);
      v28 = nb;
      v27 = v30;
      if ( (_BYTE)v5 && v41 )
LABEL_50:
        LOBYTE(v5) = memcmp(v27, v21, v28) == 0;
LABEL_20:
      j_j___libc_free_0(v27, v44);
      goto LABEL_21;
    }
    if ( (_BYTE)v5 && v25 )
      goto LABEL_50;
    if ( v27 )
      goto LABEL_20;
LABEL_21:
    if ( v21 )
      j_j___libc_free_0(v21, v43);
    if ( (_BYTE)v5 )
    {
      *v45 = v16;
      goto LABEL_25;
    }
    *(__m128i *)v59 = _mm_loadu_si128((const __m128i *)v16);
    sub_D78650(&v59[16], (const void **)v16 + 2, v28);
    if ( !*(_QWORD *)v59 && *(_OWORD *)&v59[8] == __PAIR128__(v60, -1) )
      break;
    if ( *(_QWORD *)&v59[16] )
      j_j___libc_free_0(*(_QWORD *)&v59[16], v61 - *(_QWORD *)&v59[16]);
    *(__m128i *)v59 = _mm_loadu_si128(&v52);
    sub_D78650(&v59[16], v53, v31);
    v55 = _mm_loadu_si128((const __m128i *)v16);
    sub_D78650(&s1, (const void **)v16 + 2, v32);
    v33 = s1;
    v34 = v55.m128i_i64[0] == *(_QWORD *)v59 && v55.m128i_i64[1] == *(_QWORD *)&v59[8];
    if ( v34 && (v34 = 0, v12 = v57 - (_QWORD)s1, v57 - (_QWORD)s1 == v60 - *(_QWORD *)&v59[16]) )
    {
      v35 = v58;
      if ( v12 )
      {
        na = v58;
        v37 = memcmp(s1, *(const void **)&v59[16], v12);
        v35 = na;
        if ( v37 )
        {
          v34 = 0;
          v36 = na - (_QWORD)v33;
          goto LABEL_43;
        }
      }
      v34 = v42 == 0;
    }
    else
    {
      v35 = v58;
    }
    if ( !v33 )
      goto LABEL_44;
    v36 = v35 - (_QWORD)v33;
LABEL_43:
    j_j___libc_free_0(v33, v36);
LABEL_44:
    a1 = *(_DWORD **)&v59[16];
    if ( *(_QWORD *)&v59[16] )
      j_j___libc_free_0(*(_QWORD *)&v59[16], v61 - *(_QWORD *)&v59[16]);
    if ( !v34 )
      v16 = v42;
    v51 = v38 & (v50 + v51);
    v42 = v16;
    ++v50;
  }
  if ( *(_QWORD *)&v59[16] )
    j_j___libc_free_0(*(_QWORD *)&v59[16], v61 - *(_QWORD *)&v59[16]);
  if ( v42 )
    v16 = v42;
  *v45 = v16;
LABEL_25:
  if ( v53[0] )
    j_j___libc_free_0(v53[0], v54 - (unsigned __int64)v53[0]);
  return v5;
}

// Function: sub_37EAC30
// Address: 0x37eac30
//
void __fastcall sub_37EAC30(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v4; // rdi
  __int64 (*v5)(); // rcx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __m128i *v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  const __m128i *v18; // rcx
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // r13
  __int64 v21; // rax
  const __m128i *v22; // rax
  unsigned __int64 v23; // r14
  __int64 v24; // rax
  const __m128i *v25; // rcx
  unsigned __int64 v26; // rsi
  __m128i *v27; // rdx
  const __m128i *v28; // rax
  __int64 v29; // rbx
  __int64 *v30; // rax
  __int64 v31; // rcx
  __int64 *v32; // rdx
  _QWORD *v33; // rdi
  __int64 v34; // r13
  __int64 *v35; // rax
  __int64 v36; // rsi
  unsigned __int64 v37; // rdx
  unsigned __int64 *v38; // rax
  char v39; // cl
  char v40; // si
  __int64 v41; // r13
  __int64 v42; // rbx
  char *v43; // rdx
  char *v44; // rax
  char v45; // dl
  unsigned __int64 v47; // [rsp+20h] [rbp-120h]
  unsigned __int64 v48; // [rsp+28h] [rbp-118h]
  __m128i v49; // [rsp+30h] [rbp-110h] BYREF
  char v50; // [rsp+40h] [rbp-100h]
  __m128i v51; // [rsp+50h] [rbp-F0h] BYREF
  unsigned __int64 v52; // [rsp+60h] [rbp-E0h]
  char *v53; // [rsp+68h] [rbp-D8h]
  unsigned __int64 v54; // [rsp+70h] [rbp-D0h] BYREF
  const __m128i *v55; // [rsp+78h] [rbp-C8h]
  const __m128i *v56; // [rsp+80h] [rbp-C0h]
  const __m128i *v57; // [rsp+98h] [rbp-A8h]
  const __m128i *v58; // [rsp+A0h] [rbp-A0h]
  __int64 v59; // [rsp+B0h] [rbp-90h] BYREF
  char *v60; // [rsp+B8h] [rbp-88h]
  __int64 v61; // [rsp+C0h] [rbp-80h]
  int v62; // [rsp+C8h] [rbp-78h]
  char v63; // [rsp+CCh] [rbp-74h]
  char v64; // [rsp+D0h] [rbp-70h] BYREF

  v2 = a2;
  a1[25] = a2;
  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 128LL);
  v6 = 0;
  if ( v5 != sub_2DAC790 )
  {
    v6 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v5)(v4, a2, a2);
    v2 = a1[25];
  }
  a1[26] = v6;
  v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v2 + 16) + 200LL))(*(_QWORD *)(v2 + 16));
  v8 = (__int64 *)a1[1];
  a1[27] = v7;
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
LABEL_76:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_5051514 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_76;
  }
  a1[79] = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(
             *(_QWORD *)(v9 + 8),
             &unk_5051514);
  sub_2F5FFA0(a1 + 28, a2);
  v63 = 1;
  v11 = &v54;
  v60 = &v64;
  v59 = 0;
  v61 = 8;
  v62 = 0;
  v51.m128i_i64[0] = a2;
  sub_3005920((__int64 *)&v54, &v51, (__int64)&v59, v12, v13, v14);
  v18 = v56;
  v19 = (unsigned __int64)v55;
  v52 = 0;
  v51 = (__m128i)v54;
  v53 = 0;
  v20 = (char *)v56 - (char *)v55;
  if ( v56 == v55 )
  {
    v11 = 0;
  }
  else
  {
    if ( v20 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_75;
    v21 = sub_22077B0((char *)v56 - (char *)v55);
    v18 = v56;
    v19 = (unsigned __int64)v55;
    v11 = (unsigned __int64 *)v21;
  }
  v51.m128i_i64[1] = (__int64)v11;
  v52 = (unsigned __int64)v11;
  v53 = (char *)v11 + v20;
  if ( v18 == (const __m128i *)v19 )
  {
    v23 = (unsigned __int64)v11;
  }
  else
  {
    v15 = (__m128i *)v11;
    v22 = (const __m128i *)v19;
    do
    {
      if ( v15 )
      {
        *v15 = _mm_loadu_si128(v22);
        v16 = v22[1].m128i_i64[0];
        v15[1].m128i_i64[0] = v16;
      }
      v22 = (const __m128i *)((char *)v22 + 24);
      v15 = (__m128i *)((char *)v15 + 24);
    }
    while ( v22 != v18 );
    v23 = (unsigned __int64)&v11[(((unsigned __int64)&v22[-2].m128i_u64[1] - v19) >> 3) + 3];
  }
  v19 = (unsigned __int64)v57;
  v52 = v23;
  if ( v58 == v57 )
  {
    v47 = 0;
    goto LABEL_68;
  }
  if ( (unsigned __int64)((char *)v58 - (char *)v57) > 0x7FFFFFFFFFFFFFF8LL )
LABEL_75:
    sub_4261EA(v11, v19, v15);
  v24 = sub_22077B0((char *)v58 - (char *)v57);
  v25 = v58;
  v26 = (unsigned __int64)v57;
  v47 = v24;
  v11 = (unsigned __int64 *)v51.m128i_i64[1];
  v23 = v52;
  if ( v57 == v58 )
  {
LABEL_68:
    v48 = 0;
    goto LABEL_24;
  }
  v27 = (__m128i *)v24;
  v28 = v57;
  do
  {
    if ( v27 )
    {
      *v27 = _mm_loadu_si128(v28);
      v17 = v28[1].m128i_i64[0];
      v27[1].m128i_i64[0] = v17;
    }
    v28 = (const __m128i *)((char *)v28 + 24);
    v27 = (__m128i *)((char *)v27 + 24);
  }
  while ( v28 != v25 );
  v48 = 8 * (((unsigned __int64)&v28[-2].m128i_u64[1] - v26) >> 3) + 24;
LABEL_24:
  if ( v23 - (_QWORD)v11 == v48 )
    goto LABEL_35;
  while ( 1 )
  {
    do
    {
      v29 = *(_QWORD *)(v23 - 24);
      if ( *(_BYTE *)(v23 - 8) )
        goto LABEL_32;
      v30 = *(__int64 **)(v29 + 112);
      *(_BYTE *)(v23 - 8) = 1;
      *(_QWORD *)(v23 - 16) = v30;
      v31 = *(unsigned int *)(v29 + 120);
      if ( v30 != (__int64 *)(*(_QWORD *)(v29 + 112) + 8 * v31) )
      {
        while ( 1 )
        {
          v32 = v30 + 1;
          *(_QWORD *)(v23 - 16) = v30 + 1;
          v33 = (_QWORD *)v51.m128i_i64[0];
          v34 = *v30;
          if ( !*(_BYTE *)(v51.m128i_i64[0] + 28) )
            goto LABEL_63;
          v35 = *(__int64 **)(v51.m128i_i64[0] + 8);
          v36 = *(unsigned int *)(v51.m128i_i64[0] + 20);
          v32 = &v35[v36];
          if ( v35 == v32 )
            break;
          while ( v34 != *v35 )
          {
            if ( v32 == ++v35 )
              goto LABEL_65;
          }
LABEL_32:
          v31 = *(unsigned int *)(v29 + 120);
          v30 = *(__int64 **)(v23 - 16);
          if ( v30 == (__int64 *)(*(_QWORD *)(v29 + 112) + 8 * v31) )
            goto LABEL_33;
        }
LABEL_65:
        if ( (unsigned int)v36 < *(_DWORD *)(v51.m128i_i64[0] + 16) )
        {
          *(_DWORD *)(v51.m128i_i64[0] + 20) = v36 + 1;
          *v32 = v34;
          ++*v33;
LABEL_64:
          v49.m128i_i64[0] = v34;
          v50 = 0;
          sub_37EABF0(&v51.m128i_u64[1], &v49);
          v11 = (unsigned __int64 *)v51.m128i_i64[1];
          v23 = v52;
          goto LABEL_24;
        }
LABEL_63:
        sub_C8CC70(v51.m128i_i64[0], v34, (__int64)v32, v31, v16, v17);
        if ( v45 )
          goto LABEL_64;
        goto LABEL_32;
      }
LABEL_33:
      v52 -= 24LL;
      v11 = (unsigned __int64 *)v51.m128i_i64[1];
      v23 = v52;
    }
    while ( v52 != v51.m128i_i64[1] || v52 - v51.m128i_i64[1] != v48 );
LABEL_35:
    if ( v11 == (unsigned __int64 *)v23 )
      break;
    v37 = v47;
    v38 = v11;
    while ( *v38 == *(_QWORD *)v37 )
    {
      v39 = *((_BYTE *)v38 + 16);
      if ( v39 != *(_BYTE *)(v37 + 16) || v39 && v38[1] != *(_QWORD *)(v37 + 8) )
        break;
      v38 += 3;
      v37 += 24LL;
      if ( (unsigned __int64 *)v23 == v38 )
        goto LABEL_42;
    }
  }
LABEL_42:
  if ( v47 )
  {
    j_j___libc_free_0(v47);
    v11 = (unsigned __int64 *)v51.m128i_i64[1];
  }
  if ( v11 )
    j_j___libc_free_0((unsigned __int64)v11);
  if ( v57 )
    j_j___libc_free_0((unsigned __int64)v57);
  if ( v55 )
    j_j___libc_free_0((unsigned __int64)v55);
  v40 = v63;
  v41 = *(_QWORD *)(a2 + 328);
  v42 = a2 + 320;
  if ( v41 != a2 + 320 )
  {
    while ( !v40 )
    {
      if ( sub_C8CA60((__int64)&v59, v41) )
        goto LABEL_57;
LABEL_58:
      v40 = v63;
LABEL_59:
      v41 = *(_QWORD *)(v41 + 8);
      if ( v42 == v41 )
        goto LABEL_60;
    }
    v43 = &v60[8 * HIDWORD(v61)];
    while ( 1 )
    {
      v44 = v60;
      if ( v60 != v43 )
        break;
      v41 = *(_QWORD *)(v41 + 8);
      if ( v42 == v41 )
        return;
    }
    while ( v41 != *(_QWORD *)v44 )
    {
      v44 += 8;
      if ( v43 == v44 )
        goto LABEL_59;
    }
LABEL_57:
    sub_37EAB50((__int64)a1, v41);
    goto LABEL_58;
  }
LABEL_60:
  if ( !v40 )
    _libc_free((unsigned __int64)v60);
}

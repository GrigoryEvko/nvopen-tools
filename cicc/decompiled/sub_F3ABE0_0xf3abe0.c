// Function: sub_F3ABE0
// Address: 0xf3abe0
//
__int64 __fastcall sub_F3ABE0(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rdi
  const __m128i *v7; // rdx
  const __m128i *v8; // rax
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  __m128i *v11; // r14
  __m128i *v12; // rax
  __m128i *v13; // rax
  const __m128i *v14; // rdx
  signed __int64 v15; // r13
  __m128i *v16; // rcx
  __int64 v17; // r12
  __int64 v18; // r13
  unsigned __int64 v19; // rax
  int v20; // edx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  int v23; // eax
  unsigned int v24; // esi
  __int64 v25; // rdi
  __int64 *v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rbx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 *v31; // rax
  __m128i *v32; // rdx
  __int64 *v33; // rax
  char v34; // cl
  __int64 v35; // r12
  __int64 v36; // rbx
  char *v37; // rax
  char *v38; // rdx
  __int64 *v39; // rdi
  __int64 v40; // rsi
  char v42; // dl
  __m128i *v43; // rsi
  signed __int64 v47; // [rsp+28h] [rbp-128h]
  __m128i *v48; // [rsp+30h] [rbp-120h]
  __int64 v49; // [rsp+38h] [rbp-118h]
  __int64 v50; // [rsp+40h] [rbp-110h]
  __int64 *v51; // [rsp+48h] [rbp-108h] BYREF
  __int64 *v52; // [rsp+50h] [rbp-100h]
  char *v53; // [rsp+58h] [rbp-F8h]
  __m128i v54; // [rsp+60h] [rbp-F0h] BYREF
  char v55; // [rsp+78h] [rbp-D8h]
  __m128i *v56; // [rsp+80h] [rbp-D0h] BYREF
  __m128i *v57; // [rsp+88h] [rbp-C8h]
  __m128i *v58; // [rsp+90h] [rbp-C0h]
  __int64 v59; // [rsp+98h] [rbp-B8h]
  const __m128i *v60; // [rsp+A8h] [rbp-A8h]
  const __m128i *v61; // [rsp+B0h] [rbp-A0h]
  __int64 v62; // [rsp+B8h] [rbp-98h]
  __int64 v63; // [rsp+C0h] [rbp-90h] BYREF
  char *v64; // [rsp+C8h] [rbp-88h]
  __int64 v65; // [rsp+D0h] [rbp-80h]
  int v66; // [rsp+D8h] [rbp-78h]
  char v67; // [rsp+DCh] [rbp-74h]
  char v68; // [rsp+E0h] [rbp-70h] BYREF

  v64 = &v68;
  v54.m128i_i64[0] = a1;
  v6 = (__int64 *)&v56;
  v63 = 0;
  v65 = 8;
  v66 = 0;
  v67 = 1;
  sub_F3AA10((__int64 *)&v56, &v54, (__int64)&v63, a4, a5, a6);
  v7 = v57;
  v51 = 0;
  v52 = 0;
  v50 = (__int64)v56;
  v8 = v58;
  v53 = 0;
  v9 = (char *)v58 - (char *)v57;
  if ( v58 == v57 )
  {
    v6 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_83;
    v10 = sub_22077B0((char *)v58 - (char *)v57);
    v7 = v57;
    v6 = (__int64 *)v10;
    v8 = v58;
  }
  v51 = v6;
  v52 = v6;
  v53 = (char *)v6 + v9;
  if ( v8 == v7 )
  {
    v11 = (__m128i *)v6;
  }
  else
  {
    v11 = (__m128i *)((char *)v6 + (char *)v8 - (char *)v7);
    v12 = (__m128i *)v6;
    do
    {
      if ( v12 )
      {
        *v12 = _mm_loadu_si128(v7);
        v12[1] = _mm_loadu_si128(v7 + 1);
      }
      v12 += 2;
      v7 += 2;
    }
    while ( v12 != v11 );
  }
  v7 = v60;
  v52 = (__int64 *)v11;
  v47 = (char *)v61 - (char *)v60;
  if ( v61 == v60 )
  {
    v48 = 0;
    goto LABEL_73;
  }
  if ( (unsigned __int64)((char *)v61 - (char *)v60) > 0x7FFFFFFFFFFFFFE0LL )
LABEL_83:
    sub_4261EA(v6, &v54, v7);
  v13 = (__m128i *)sub_22077B0((char *)v61 - (char *)v60);
  v14 = v60;
  v6 = v51;
  v48 = v13;
  v11 = (__m128i *)v52;
  if ( v61 == v60 )
  {
LABEL_73:
    v49 = 0;
    goto LABEL_17;
  }
  v15 = (char *)v61 - (char *)v60;
  v16 = (__m128i *)((char *)v13 + (char *)v61 - (char *)v60);
  do
  {
    if ( v13 )
    {
      *v13 = _mm_loadu_si128(v14);
      v13[1] = _mm_loadu_si128(v14 + 1);
    }
    v13 += 2;
    v14 += 2;
  }
  while ( v13 != v16 );
  v49 = v15;
  while ( 1 )
  {
LABEL_17:
    if ( (char *)v11 - (char *)v6 != v49 )
      goto LABEL_18;
LABEL_35:
    if ( v11 == (__m128i *)v6 )
      break;
    v32 = v48;
    v33 = v6;
    while ( *v33 == v32->m128i_i64[0] )
    {
      v34 = *((_BYTE *)v33 + 24);
      if ( v34 != v32[1].m128i_i8[8] || v34 && *((_DWORD *)v33 + 4) != v32[1].m128i_i32[0] )
        break;
      v33 += 4;
      v32 += 2;
      if ( v11 == (__m128i *)v33 )
        goto LABEL_42;
    }
LABEL_18:
    while ( 2 )
    {
      v17 = v11[-2].m128i_i64[0];
      v18 = v17 + 48;
      if ( !v11[-1].m128i_i8[8] )
      {
        v19 = *(_QWORD *)(v17 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v18 == v19 )
        {
          v21 = 0;
        }
        else
        {
          if ( !v19 )
            BUG();
          v20 = *(unsigned __int8 *)(v19 - 24);
          v21 = v19 - 24;
          if ( (unsigned int)(v20 - 30) >= 0xB )
            v21 = 0;
        }
        v11[-2].m128i_i64[1] = v21;
        v11[-1].m128i_i32[0] = 0;
        v11[-1].m128i_i8[8] = 1;
      }
LABEL_24:
      v22 = *(_QWORD *)(v17 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v22 == v18 )
        goto LABEL_67;
LABEL_25:
      if ( !v22 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v22 - 24) - 30 <= 0xA )
      {
        v23 = sub_B46E30(v22 - 24);
        v24 = v11[-1].m128i_u32[0];
        if ( v24 == v23 )
          goto LABEL_68;
        goto LABEL_28;
      }
LABEL_67:
      while ( 1 )
      {
        v24 = v11[-1].m128i_u32[0];
        if ( !v24 )
          break;
LABEL_28:
        v25 = v11[-2].m128i_i64[1];
        v11[-1].m128i_i32[0] = v24 + 1;
        v28 = sub_B46EC0(v25, v24);
        if ( *(_BYTE *)(v50 + 28) )
        {
          v31 = *(__int64 **)(v50 + 8);
          v27 = *(unsigned int *)(v50 + 20);
          v26 = &v31[v27];
          if ( v31 != v26 )
          {
            while ( v28 != *v31 )
            {
              if ( v26 == ++v31 )
                goto LABEL_32;
            }
            goto LABEL_24;
          }
LABEL_32:
          if ( (unsigned int)v27 < *(_DWORD *)(v50 + 16) )
          {
            *(_DWORD *)(v50 + 20) = v27 + 1;
            *v26 = v28;
            ++*(_QWORD *)v50;
LABEL_34:
            v54.m128i_i64[0] = v28;
            v55 = 0;
            sub_F3A9D0((__int64)&v51, &v54);
            v11 = (__m128i *)v52;
            v6 = v51;
            if ( (char *)v52 - (char *)v51 != v49 )
              goto LABEL_18;
            goto LABEL_35;
          }
        }
        sub_C8CC70(v50, v28, (__int64)v26, v27, v29, v30);
        if ( v42 )
          goto LABEL_34;
        v22 = *(_QWORD *)(v17 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v22 != v18 )
          goto LABEL_25;
      }
LABEL_68:
      v52 -= 4;
      v6 = v51;
      v11 = (__m128i *)v52;
      if ( v52 != v51 )
        continue;
      break;
    }
  }
LABEL_42:
  if ( v48 )
  {
    j_j___libc_free_0(v48, v47);
    v6 = v51;
  }
  if ( v6 )
    j_j___libc_free_0(v6, v53 - (char *)v6);
  if ( v60 )
    j_j___libc_free_0(v60, v62 - (_QWORD)v60);
  if ( v57 )
    j_j___libc_free_0(v57, v59 - (_QWORD)v57);
  v56 = 0;
  v57 = 0;
  v35 = *(_QWORD *)(a1 + 80);
  v58 = 0;
  if ( a1 + 72 != v35 )
  {
    while ( 1 )
    {
      v36 = v35 - 24;
      if ( !v35 )
        v36 = 0;
      if ( v67 )
        break;
      if ( !sub_C8CA60((__int64)&v63, v36) )
        goto LABEL_77;
LABEL_58:
      v35 = *(_QWORD *)(v35 + 8);
      if ( a1 + 72 == v35 )
      {
        v39 = (__int64 *)v56;
        v40 = ((char *)v57 - (char *)v56) >> 3;
        goto LABEL_60;
      }
    }
    v37 = v64;
    v38 = &v64[8 * HIDWORD(v65)];
    if ( v64 != v38 )
    {
      while ( v36 != *(_QWORD *)v37 )
      {
        v37 += 8;
        if ( v38 == v37 )
          goto LABEL_77;
      }
      goto LABEL_58;
    }
LABEL_77:
    v54.m128i_i64[0] = v36;
    v43 = v57;
    if ( v57 == v58 )
    {
      sub_F38A10((__int64)&v56, v57, &v54);
    }
    else
    {
      if ( v57 )
      {
        v57->m128i_i64[0] = v36;
        v43 = v57;
      }
      v57 = (__m128i *)&v43->m128i_u64[1];
    }
    goto LABEL_58;
  }
  v40 = 0;
  v39 = 0;
LABEL_60:
  sub_F344A0(v39, v40, a2, a3);
  LOBYTE(v35) = v57 != v56;
  if ( v56 )
  {
    v40 = (char *)v58 - (char *)v56;
    j_j___libc_free_0(v56, (char *)v58 - (char *)v56);
  }
  if ( !v67 )
    _libc_free(v64, v40);
  return (unsigned int)v35;
}

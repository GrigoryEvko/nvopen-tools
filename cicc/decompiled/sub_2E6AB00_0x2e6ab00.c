// Function: sub_2E6AB00
// Address: 0x2e6ab00
//
void __fastcall sub_2E6AB00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  bool v7; // zf
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rdx
  __m128i v14; // xmm0
  __int64 v15; // rdx
  char **v16; // rsi
  unsigned __int64 *v17; // rdi
  unsigned __int64 v18; // rax
  __m128i *v19; // rdx
  const __m128i *v20; // r12
  unsigned __int64 v21; // r10
  unsigned __int64 v22; // r12
  unsigned __int64 *v23; // rsi
  char *v24; // r14
  __int64 v25; // rsi
  char *v26; // r15
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  char *v29; // r15
  __int64 v30; // rsi
  char *v31; // r14
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  __int64 v34; // [rsp+8h] [rbp-528h]
  __m128i v35; // [rsp+10h] [rbp-520h] BYREF
  unsigned __int64 v36; // [rsp+20h] [rbp-510h]
  __int64 v37; // [rsp+28h] [rbp-508h]
  unsigned __int64 *v38; // [rsp+30h] [rbp-500h] BYREF
  __int64 v39; // [rsp+38h] [rbp-4F8h]
  _BYTE v40[512]; // [rsp+40h] [rbp-4F0h] BYREF
  char **v41; // [rsp+240h] [rbp-2F0h] BYREF
  __int64 v42; // [rsp+248h] [rbp-2E8h]
  char *v43; // [rsp+250h] [rbp-2E0h] BYREF
  unsigned int v44; // [rsp+258h] [rbp-2D8h]
  char v45; // [rsp+370h] [rbp-1C0h] BYREF
  char v46; // [rsp+378h] [rbp-1B8h]
  char *v47; // [rsp+380h] [rbp-1B0h] BYREF
  unsigned int v48; // [rsp+388h] [rbp-1A8h]
  char v49; // [rsp+4A0h] [rbp-90h] BYREF
  char *v50; // [rsp+4A8h] [rbp-88h]
  char v51; // [rsp+4B8h] [rbp-78h] BYREF

  v6 = *(_QWORD *)(a1 + 544);
  v7 = *(_BYTE *)(a1 + 560) == 1;
  v37 = v6;
  if ( v7 && v6 )
  {
    while ( 1 )
    {
      v9 = *(unsigned int *)(a1 + 8);
      v10 = *(_QWORD *)(a1 + 528);
      if ( v10 == v9 )
        return;
      v11 = *(_QWORD *)a1 + 32 * v10;
      v12 = *(_QWORD *)a1 + 32 * v9;
      if ( !*(_BYTE *)v11 )
        break;
      v41 = &v43;
      v42 = 0x200000000LL;
      if ( v11 == v12 )
      {
        v15 = 0;
        v16 = &v43;
      }
      else
      {
        v18 = v11 + 8;
        v16 = &v43;
        v15 = 0;
        while ( 1 )
        {
          v20 = (const __m128i *)v18;
          if ( !*(_BYTE *)(v18 - 8) )
            break;
          v21 = v15 + 1;
          if ( v15 + 1 > (unsigned __int64)HIDWORD(v42) )
          {
            if ( (unsigned __int64)v16 > v18 || (unsigned __int64)&v16[3 * v15] <= v18 )
            {
              v36 = v18;
              v35.m128i_i64[0] = v12;
              sub_C8D5F0((__int64)&v41, &v43, v21, 0x18u, v12, a6);
              v16 = v41;
              v15 = (unsigned int)v42;
              v12 = v35.m128i_i64[0];
              v18 = v36;
            }
            else
            {
              v22 = v18 - (_QWORD)v16;
              v36 = v12;
              v35.m128i_i64[0] = v18;
              sub_C8D5F0((__int64)&v41, &v43, v21, 0x18u, v12, a6);
              v16 = v41;
              v15 = (unsigned int)v42;
              v12 = v36;
              v18 = v35.m128i_i64[0];
              v20 = (const __m128i *)((char *)v41 + v22);
            }
          }
          v19 = (__m128i *)&v16[3 * v15];
          *v19 = _mm_loadu_si128(v20);
          v19[1].m128i_i64[0] = v20[1].m128i_i64[0];
          a4 = v18 + 32;
          v16 = v41;
          v15 = (unsigned int)(v42 + 1);
          LODWORD(v42) = v42 + 1;
          if ( v12 == v18 + 24 )
            break;
          v18 += 32LL;
        }
      }
      sub_2E65A90(a1, v16, v15, a4, v12, a6);
      v17 = (unsigned __int64 *)v41;
      *(_QWORD *)(a1 + 528) += (unsigned int)v42;
      if ( v17 == (unsigned __int64 *)&v43 )
        goto LABEL_14;
LABEL_13:
      _libc_free((unsigned __int64)v17);
LABEL_14:
      if ( !*(_QWORD *)(a1 + 544) )
        return;
    }
    v13 = 0;
    v38 = (unsigned __int64 *)v40;
    v39 = 0x2000000000LL;
    if ( v11 == v12 )
    {
      v23 = (unsigned __int64 *)v40;
    }
    else
    {
      do
      {
        if ( *(_BYTE *)v11 )
          break;
        v14 = _mm_loadu_si128((const __m128i *)(v11 + 8));
        if ( v13 + 1 > (unsigned __int64)HIDWORD(v39) )
        {
          v34 = v12;
          v36 = v11;
          v35 = v14;
          sub_C8D5F0((__int64)&v38, v40, v13 + 1, 0x10u, v12, v13 + 1);
          v13 = (unsigned int)v39;
          v12 = v34;
          v11 = v36;
          v14 = _mm_load_si128(&v35);
        }
        v11 += 32;
        *(__m128i *)&v38[2 * v13] = v14;
        v13 = (unsigned int)(v39 + 1);
        LODWORD(v39) = v39 + 1;
      }
      while ( v11 != v12 );
      v23 = v38;
    }
    sub_2E6A250((__int64)&v41, v23, v13, 1u);
    sub_2E73D70(v37, &v41, 0);
    if ( v50 != &v51 )
      _libc_free((unsigned __int64)v50);
    if ( (v46 & 1) != 0 )
    {
      v26 = &v49;
      v24 = (char *)&v47;
    }
    else
    {
      v24 = v47;
      v25 = 72LL * v48;
      if ( !v48 )
        goto LABEL_56;
      v26 = &v47[v25];
      if ( &v47[v25] == v47 )
        goto LABEL_56;
    }
    do
    {
      if ( *(_QWORD *)v24 != -4096 && *(_QWORD *)v24 != -8192 )
      {
        v27 = *((_QWORD *)v24 + 5);
        if ( (char *)v27 != v24 + 56 )
          _libc_free(v27);
        v28 = *((_QWORD *)v24 + 1);
        if ( (char *)v28 != v24 + 24 )
          _libc_free(v28);
      }
      v24 += 72;
    }
    while ( v26 != v24 );
    if ( (v46 & 1) != 0 )
    {
      if ( (v42 & 1) == 0 )
      {
LABEL_41:
        v29 = v43;
        v30 = 72LL * v44;
        if ( !v44 )
          goto LABEL_54;
        v31 = &v43[v30];
        if ( &v43[v30] == v43 )
          goto LABEL_54;
        goto LABEL_43;
      }
LABEL_57:
      v31 = &v45;
      v29 = (char *)&v43;
      do
      {
LABEL_43:
        if ( *(_QWORD *)v29 != -8192 && *(_QWORD *)v29 != -4096 )
        {
          v32 = *((_QWORD *)v29 + 5);
          if ( (char *)v32 != v29 + 56 )
            _libc_free(v32);
          v33 = *((_QWORD *)v29 + 1);
          if ( (char *)v33 != v29 + 24 )
            _libc_free(v33);
        }
        v29 += 72;
      }
      while ( v29 != v31 );
      if ( (v42 & 1) != 0 )
      {
LABEL_51:
        v17 = v38;
        *(_QWORD *)(a1 + 528) += (unsigned int)v39;
        if ( v17 == (unsigned __int64 *)v40 )
          goto LABEL_14;
        goto LABEL_13;
      }
      v29 = v43;
      v30 = 72LL * v44;
LABEL_54:
      sub_C7D6A0((__int64)v29, v30, 8);
      goto LABEL_51;
    }
    v24 = v47;
    v25 = 72LL * v48;
LABEL_56:
    sub_C7D6A0((__int64)v24, v25, 8);
    if ( (v42 & 1) == 0 )
      goto LABEL_41;
    goto LABEL_57;
  }
}

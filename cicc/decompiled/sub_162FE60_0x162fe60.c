// Function: sub_162FE60
// Address: 0x162fe60
//
void __fastcall sub_162FE60(
        const __m128i *a1,
        unsigned __int8 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int32 v12; // eax
  const __m128i *v13; // r13
  const __m128i *v14; // r15
  __m128i *v15; // r13
  const __m128i *v16; // rcx
  __int64 v17; // r8
  const __m128i *v18; // rdx
  __m128 *v19; // rcx
  const __m128i *v20; // rax
  __int64 v21; // r15
  __m128i *v22; // r12
  unsigned __int64 v23; // rax
  double v24; // xmm4_8
  double v25; // xmm5_8
  char *v26; // r12
  const __m128i *v27; // rdi
  int v28; // esi
  unsigned __int8 **v29; // r8
  int v30; // r9d
  unsigned int v31; // eax
  unsigned __int8 **v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // rdi
  __m128i *v35; // r15
  __m128i *v36; // rdi
  __int32 v37; // eax
  const __m128i *v38; // r8
  int v39; // edx
  unsigned int v40; // eax
  __int64 *v41; // rdi
  __int64 v42; // r9
  unsigned __int32 v43; // eax
  __int32 v44; // edx
  int v45; // edi
  int v46; // r10d
  int v47; // [rsp+0h] [rbp-110h]
  __m128i *v48; // [rsp+10h] [rbp-100h] BYREF
  __int64 v49; // [rsp+18h] [rbp-F8h]
  _BYTE v50[240]; // [rsp+20h] [rbp-F0h] BYREF

  v12 = (unsigned __int32)a1[1].m128i_i32[2] >> 1;
  if ( (a1[1].m128i_i8[8] & 1) != 0 )
  {
    v13 = a1 + 8;
    v14 = a1 + 2;
    if ( !v12 )
      goto LABEL_7;
  }
  else
  {
    v14 = (const __m128i *)a1[2].m128i_i64[0];
    v13 = (const __m128i *)((char *)v14 + 24 * a1[2].m128i_u32[2]);
    if ( !v12 )
      goto LABEL_7;
  }
  if ( v14 == v13 )
    goto LABEL_7;
  while ( v14->m128i_i64[0] == -8 || v14->m128i_i64[0] == -4 )
  {
    v14 = (const __m128i *)((char *)v14 + 24);
    if ( v13 == v14 )
      goto LABEL_7;
  }
  v48 = (__m128i *)v50;
  v49 = 0x800000000LL;
  if ( v13 == v14 )
  {
LABEL_7:
    v15 = (__m128i *)v50;
    goto LABEL_8;
  }
  v16 = v14;
  v17 = 0;
  while ( 1 )
  {
    v18 = (const __m128i *)((char *)v16 + 24);
    if ( v13 == (const __m128i *)&v16[1].m128i_u64[1] )
      break;
    while ( 1 )
    {
      v16 = v18;
      if ( v18->m128i_i64[0] != -8 && v18->m128i_i64[0] != -4 )
        break;
      v18 = (const __m128i *)((char *)v18 + 24);
      if ( v13 == v18 )
        goto LABEL_17;
    }
    ++v17;
    if ( v13 == v18 )
      goto LABEL_18;
  }
LABEL_17:
  ++v17;
LABEL_18:
  v19 = (__m128 *)v50;
  if ( v17 > 8 )
  {
    v47 = v17;
    sub_16CD150(&v48, v50, v17, 24);
    LODWORD(v17) = v47;
    v19 = (__m128 *)((char *)v48 + 24 * (unsigned int)v49);
  }
  do
  {
    if ( v19 )
    {
      a3 = (__m128)_mm_loadu_si128(v14);
      *v19 = a3;
      v19[1].m128_u64[0] = v14[1].m128i_u64[0];
    }
    v20 = (const __m128i *)((char *)v14 + 24);
    if ( v13 == (const __m128i *)&v14[1].m128i_u64[1] )
      break;
    while ( 1 )
    {
      v14 = v20;
      if ( v20->m128i_i64[0] != -4 && v20->m128i_i64[0] != -8 )
        break;
      v20 = (const __m128i *)((char *)v20 + 24);
      if ( v13 == v20 )
        goto LABEL_26;
    }
    v19 = (__m128 *)((char *)v19 + 24);
  }
  while ( v20 != v13 );
LABEL_26:
  v15 = v48;
  LODWORD(v49) = v49 + v17;
  v21 = 24LL * (unsigned int)v49;
  v22 = (__m128i *)((char *)v48 + v21);
  if ( &v48->m128i_i8[v21] != (__int8 *)v48 )
  {
    _BitScanReverse64(&v23, 0xAAAAAAAAAAAAAAABLL * (v21 >> 3));
    sub_161D320(
      v48->m128i_i8,
      (__m128i *)((char *)v48 + v21),
      2LL * (int)(63 - (v23 ^ 0x3F)),
      (__int64)v19,
      (unsigned int)v49);
    if ( (unsigned __int64)v21 > 0x180 )
    {
      v35 = v15 + 24;
      sub_161CED0(v15, v15 + 24);
      if ( v22 != &v15[24] )
      {
        do
        {
          v36 = v35;
          v35 = (__m128i *)((char *)v35 + 24);
          sub_161CE70(v36);
        }
        while ( v22 != v35 );
      }
    }
    else
    {
      sub_161CED0(v15, v22);
    }
    v15 = v48;
    v26 = &v48->m128i_i8[24 * (unsigned int)v49];
    if ( v26 != (char *)v48 )
    {
      while ( 1 )
      {
        if ( (a1[1].m128i_i8[8] & 1) != 0 )
        {
          v27 = a1 + 2;
          v28 = 3;
        }
        else
        {
          v37 = a1[2].m128i_i32[2];
          v27 = (const __m128i *)a1[2].m128i_i64[0];
          if ( !v37 )
            goto LABEL_42;
          v28 = v37 - 1;
        }
        v29 = (unsigned __int8 **)v15->m128i_i64[0];
        v30 = 1;
        v31 = v28 & (((unsigned int)v15->m128i_i64[0] >> 9) ^ ((unsigned int)v15->m128i_i64[0] >> 4));
        v32 = (unsigned __int8 **)v27->m128i_i64[3 * v31];
        if ( (unsigned __int8 **)v15->m128i_i64[0] != v32 )
        {
          while ( v32 != (unsigned __int8 **)-4LL )
          {
            v31 = v28 & (v30 + v31);
            v32 = (unsigned __int8 **)v27->m128i_i64[3 * v31];
            if ( v29 == v32 )
              goto LABEL_33;
            ++v30;
          }
          goto LABEL_42;
        }
LABEL_33:
        v33 = v15->m128i_i64[1];
        v34 = v33 & 0xFFFFFFFFFFFFFFFCLL;
        if ( (v33 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
        {
          if ( (v33 & 2) != 0 )
            sub_162FCD0(v34, v15->m128i_i64[0], a2, *(double *)a3.m128_u64, a4, a5, a6, v24, v25, a9, a10);
          else
            sub_16290E0(v34, (__int64)a2);
        }
        else
        {
          *v29 = a2;
          if ( a2 )
            sub_1623A60((__int64)v29, (__int64)a2, 2);
          if ( (a1[1].m128i_i8[8] & 1) != 0 )
          {
            v38 = a1 + 2;
            v39 = 3;
            goto LABEL_57;
          }
          v44 = a1[2].m128i_i32[2];
          v38 = (const __m128i *)a1[2].m128i_i64[0];
          if ( v44 )
          {
            v39 = v44 - 1;
LABEL_57:
            v40 = v39 & (((unsigned int)v15->m128i_i64[0] >> 9) ^ ((unsigned int)v15->m128i_i64[0] >> 4));
            v41 = &v38->m128i_i64[3 * v40];
            v42 = *v41;
            if ( *v41 == v15->m128i_i64[0] )
            {
LABEL_58:
              *v41 = -8;
              v43 = a1[1].m128i_u32[2];
              ++a1[1].m128i_i32[3];
              a1[1].m128i_i32[2] = (2 * (v43 >> 1) - 2) | v43 & 1;
            }
            else
            {
              v45 = 1;
              while ( v42 != -4 )
              {
                v46 = v45 + 1;
                v40 = v39 & (v45 + v40);
                v41 = &v38->m128i_i64[3 * v40];
                v42 = *v41;
                if ( v15->m128i_i64[0] == *v41 )
                  goto LABEL_58;
                v45 = v46;
              }
            }
          }
        }
LABEL_42:
        v15 = (__m128i *)((char *)v15 + 24);
        if ( v26 == (char *)v15 )
        {
          v15 = v48;
          break;
        }
      }
    }
  }
LABEL_8:
  if ( v15 != (__m128i *)v50 )
    _libc_free((unsigned __int64)v15);
}

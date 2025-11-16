// Function: sub_9DD2C0
// Address: 0x9dd2c0
//
__int64 *__fastcall sub_9DD2C0(__int64 *a1, unsigned __int64 a2)
{
  const __m128i *v3; // rdx
  const __m128i *v4; // rax
  const __m128i *v5; // r12
  const __m128i *v6; // r13
  const __m128i *v7; // rax
  __int64 v8; // rax
  const __m128i *v9; // rax
  const __m128i *v10; // rax
  __int64 v11; // rax
  __m128i *v12; // rsi
  unsigned int v13; // edx
  const __m128i *v14; // rbx
  char v15; // dl
  const __m128i *v16; // r14
  const __m128i *v17; // r12
  __m128i *v18; // rsi
  unsigned int v19; // edx
  const __m128i *v20; // rbx
  char v21; // dl
  char v22; // al
  __int64 v23; // rdi
  _QWORD *v24; // rsi
  char v25; // al
  __int64 v26; // rax
  __int64 v27; // rax
  const char *v28; // rax
  char v30; // dl
  unsigned __int64 v31; // rax
  const __m128i *v32; // r12
  unsigned int v33; // edx
  __int64 v34; // rsi
  const __m128i *v35; // rbx
  __int32 v36; // edx
  unsigned int v37; // edx
  __int32 v38; // edx
  unsigned int v39; // edx
  __int32 v40; // edx
  char v41; // si
  char v42; // si
  char v43; // si
  const __m128i *v44; // [rsp+8h] [rbp-B8h]
  __int64 v45; // [rsp+10h] [rbp-B0h]
  __int64 v46; // [rsp+18h] [rbp-A8h]
  __int64 v47; // [rsp+20h] [rbp-A0h]
  const __m128i *v48; // [rsp+28h] [rbp-98h]
  const __m128i *v49; // [rsp+28h] [rbp-98h]
  const __m128i *v50; // [rsp+28h] [rbp-98h]
  const __m128i *v51; // [rsp+30h] [rbp-90h]
  const __m128i **v52; // [rsp+38h] [rbp-88h]
  const __m128i *v54; // [rsp+48h] [rbp-78h]
  _QWORD *v55; // [rsp+50h] [rbp-70h] BYREF
  char v56; // [rsp+58h] [rbp-68h]
  const char *v57; // [rsp+60h] [rbp-60h] BYREF
  char v58; // [rsp+68h] [rbp-58h]
  char v59; // [rsp+80h] [rbp-40h]
  char v60; // [rsp+81h] [rbp-3Fh]

  v52 = (const __m128i **)(a2 + 1408);
  v3 = *(const __m128i **)(a2 + 1416);
  v47 = *(_QWORD *)(a2 + 1424);
  v4 = *(const __m128i **)(a2 + 1432);
  v5 = *(const __m128i **)(a2 + 1408);
  v6 = v3 - 1;
  *(_QWORD *)(a2 + 1408) = 0;
  v54 = v4;
  v7 = *(const __m128i **)(a2 + 1440);
  *(_QWORD *)(a2 + 1416) = 0;
  v48 = v7;
  v8 = *(_QWORD *)(a2 + 1448);
  *(_QWORD *)(a2 + 1424) = 0;
  v45 = v8;
  v9 = *(const __m128i **)(a2 + 1456);
  *(_QWORD *)(a2 + 1432) = 0;
  v51 = v9;
  v10 = *(const __m128i **)(a2 + 1464);
  *(_QWORD *)(a2 + 1440) = 0;
  v44 = v10;
  v11 = *(_QWORD *)(a2 + 1472);
  *(_QWORD *)(a2 + 1448) = 0;
  v46 = v11;
  *(_QWORD *)(a2 + 1456) = 0;
  *(_QWORD *)(a2 + 1464) = 0;
  *(_QWORD *)(a2 + 1472) = 0;
  if ( v3 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v13 = v6->m128i_u32[2];
        v14 = v6;
        if ( v13 < (unsigned int)((__int64)(*(_QWORD *)(a2 + 752) - *(_QWORD *)(a2 + 744)) >> 5) )
          break;
        v12 = *(__m128i **)(a2 + 1416);
        if ( v12 == *(__m128i **)(a2 + 1424) )
        {
          sub_9CBE80(v52, v12, v6);
        }
        else
        {
          if ( v12 )
          {
            *v12 = _mm_loadu_si128(v6);
            v12 = *(__m128i **)(a2 + 1416);
          }
          *(_QWORD *)(a2 + 1416) = v12 + 1;
        }
LABEL_7:
        --v6;
        if ( v5 == v14 )
          goto LABEL_14;
      }
      sub_9DD260((__int64)&v57, a2, v13);
      v15 = v58 & 1;
      v58 = (2 * (v58 & 1)) | v58 & 0xFD;
      if ( v15 )
      {
LABEL_51:
        *a1 = (unsigned __int64)v57 | 1;
        goto LABEL_43;
      }
      sub_B30160(v6->m128i_i64[0], v57);
      if ( (v58 & 2) != 0 )
LABEL_39:
        sub_9D2280(&v57);
      if ( (v58 & 1) == 0 || !v57 )
        goto LABEL_7;
      --v6;
      (*(void (__fastcall **)(const char *))(*(_QWORD *)v57 + 8LL))(v57);
    }
    while ( v5 != v14 );
  }
LABEL_14:
  v16 = v48 - 1;
  if ( v48 != v54 )
  {
    v49 = v5;
    v17 = v16;
    do
    {
      v19 = v17->m128i_u32[2];
      v20 = v17;
      if ( v19 >= (unsigned int)((__int64)(*(_QWORD *)(a2 + 752) - *(_QWORD *)(a2 + 744)) >> 5) )
      {
        v18 = *(__m128i **)(a2 + 1440);
        if ( v18 == *(__m128i **)(a2 + 1448) )
        {
          sub_9CC000((const __m128i **)(a2 + 1432), v18, v17);
        }
        else
        {
          if ( v18 )
          {
            *v18 = _mm_loadu_si128(v17);
            v18 = *(__m128i **)(a2 + 1440);
          }
          *(_QWORD *)(a2 + 1440) = v18 + 1;
        }
      }
      else
      {
        sub_9DD260((__int64)&v55, a2, v19);
        v21 = v56 & 1;
        v22 = (2 * (v56 & 1)) | v56 & 0xFD;
        v56 = v22;
        if ( v21 )
        {
          v30 = v22;
          v31 = (unsigned __int64)v55;
          v55 = 0;
          v5 = v49;
          v56 = v30 & 0xFD;
          *a1 = v31 | 1;
          goto LABEL_54;
        }
        v23 = v17->m128i_i64[0];
        v24 = v55;
        v25 = *(_BYTE *)v17->m128i_i64[0];
        if ( v25 == 1 )
        {
          if ( *(_QWORD *)(v23 + 8) != v55[1] )
          {
            v60 = 1;
            v5 = v49;
            v28 = "Alias and aliasee types don't match";
LABEL_41:
            v57 = v28;
            v59 = 3;
            sub_9C81F0(a1, a2 + 8, (__int64)&v57);
            if ( (v56 & 2) != 0 )
LABEL_56:
              sub_9D2280(&v55);
            if ( (v56 & 1) == 0 )
              goto LABEL_43;
LABEL_54:
            if ( v55 )
              (*(void (__fastcall **)(_QWORD *))(*v55 + 8LL))(v55);
            goto LABEL_43;
          }
          sub_B303B0(v23, v55);
        }
        else
        {
          if ( v25 != 2 )
          {
            v60 = 1;
            v5 = v49;
            v28 = "Expected an alias or an ifunc";
            goto LABEL_41;
          }
          if ( *(_QWORD *)(v23 - 32) )
          {
            v26 = *(_QWORD *)(v23 - 24);
            **(_QWORD **)(v23 - 16) = v26;
            if ( v26 )
              *(_QWORD *)(v26 + 16) = *(_QWORD *)(v23 - 16);
          }
          *(_QWORD *)(v23 - 32) = v24;
          if ( v24 )
          {
            v27 = v24[2];
            *(_QWORD *)(v23 - 24) = v27;
            if ( v27 )
              *(_QWORD *)(v27 + 16) = v23 - 24;
            *(_QWORD *)(v23 - 16) = v24 + 2;
            v24[2] = v23 - 32;
          }
        }
        if ( (v56 & 2) != 0 )
          goto LABEL_56;
        if ( (v56 & 1) != 0 && v55 )
          (*(void (__fastcall **)(_QWORD *))(*v55 + 8LL))(v55);
      }
      --v17;
    }
    while ( v54 != v20 );
    v5 = v49;
  }
  if ( v44 == v51 )
    goto LABEL_93;
  v50 = v5;
  v32 = v44;
  do
  {
    v35 = v32;
    v36 = v32[-1].m128i_i32[0];
    v32 = (const __m128i *)((char *)v32 - 24);
    if ( v36 )
    {
      v37 = v36 - 1;
      if ( v37 < (unsigned int)((__int64)(*(_QWORD *)(a2 + 752) - *(_QWORD *)(a2 + 744)) >> 5) )
      {
        sub_9DD260((__int64)&v57, a2, v37);
        v41 = v58 & 1;
        v58 = (2 * (v58 & 1)) | v58 & 0xFD;
        if ( v41 )
          goto LABEL_50;
        sub_B2E8C0(v32->m128i_i64[0], v57);
        v32->m128i_i32[2] = 0;
        if ( (v58 & 2) != 0 )
          goto LABEL_39;
        if ( (v58 & 1) != 0 && v57 )
          (*(void (__fastcall **)(const char *))(*(_QWORD *)v57 + 8LL))(v57);
      }
    }
    v38 = v32->m128i_i32[3];
    if ( v38 )
    {
      v39 = v38 - 1;
      if ( v39 < (unsigned int)((__int64)(*(_QWORD *)(a2 + 752) - *(_QWORD *)(a2 + 744)) >> 5) )
      {
        sub_9DD260((__int64)&v57, a2, v39);
        v42 = v58 & 1;
        v58 = (2 * (v58 & 1)) | v58 & 0xFD;
        if ( v42 )
          goto LABEL_50;
        sub_B2E9C0(v32->m128i_i64[0], v57);
        v32->m128i_i32[3] = 0;
        if ( (v58 & 2) != 0 )
          goto LABEL_39;
        if ( (v58 & 1) != 0 && v57 )
          (*(void (__fastcall **)(const char *))(*(_QWORD *)v57 + 8LL))(v57);
      }
    }
    v40 = v32[1].m128i_i32[0];
    if ( !v40 )
    {
      if ( !v35[-1].m128i_i64[0] )
        continue;
LABEL_61:
      v34 = *(_QWORD *)(a2 + 1464);
      if ( v34 == *(_QWORD *)(a2 + 1472) )
      {
        sub_9C2BD0(a2 + 1456, (_BYTE *)v34, v32);
      }
      else
      {
        if ( v34 )
        {
          *(__m128i *)v34 = _mm_loadu_si128(v32);
          *(_QWORD *)(v34 + 16) = v32[1].m128i_i64[0];
          v34 = *(_QWORD *)(a2 + 1464);
        }
        *(_QWORD *)(a2 + 1464) = v34 + 24;
      }
      continue;
    }
    v33 = v40 - 1;
    if ( v33 >= (unsigned int)((__int64)(*(_QWORD *)(a2 + 752) - *(_QWORD *)(a2 + 744)) >> 5) )
      goto LABEL_61;
    sub_9DD260((__int64)&v57, a2, v33);
    v43 = v58 & 1;
    v58 = (2 * (v58 & 1)) | v58 & 0xFD;
    if ( v43 )
    {
LABEL_50:
      v5 = v50;
      goto LABEL_51;
    }
    sub_B2EAD0(v32->m128i_i64[0], v57);
    v32[1].m128i_i32[0] = 0;
    if ( (v58 & 2) != 0 )
      goto LABEL_39;
    if ( (v58 & 1) != 0 && v57 )
      (*(void (__fastcall **)(const char *))(*(_QWORD *)v57 + 8LL))(v57);
    if ( v35[-1].m128i_i64[0] || v32[1].m128i_i32[0] )
      goto LABEL_61;
  }
  while ( v51 != v32 );
  v5 = v50;
LABEL_93:
  *a1 = 1;
LABEL_43:
  if ( v51 )
    j_j___libc_free_0(v51, v46 - (_QWORD)v51);
  if ( v54 )
    j_j___libc_free_0(v54, v45 - (_QWORD)v54);
  if ( v5 )
    j_j___libc_free_0(v5, v47 - (_QWORD)v5);
  return a1;
}

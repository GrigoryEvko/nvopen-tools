// Function: sub_20211B0
// Address: 0x20211b0
//
__int64 __fastcall sub_20211B0(const __m128i *a1, __m128i a2, __m128i a3, __m128i a4)
{
  const __m128i *v4; // rax
  _BYTE *v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r14
  unsigned __int8 *v8; // r13
  __int64 v9; // rsi
  unsigned __int8 *v10; // r15
  int v11; // r12d
  int v12; // eax
  int v13; // eax
  __int64 v14; // rdi
  int v15; // eax
  unsigned int v16; // r12d
  const __m128i *v18; // r8
  __int64 v19; // r12
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // rbx
  __int32 v23; // r14d
  __int64 v24; // rcx
  _QWORD *v25; // rdi
  __int64 v26; // rsi
  int v27; // r10d
  __int64 v28; // r9
  __int64 i; // rdx
  _QWORD *v30; // rax
  int v31; // edx
  __int64 v32; // rbx
  __int64 v33; // r14
  unsigned __int32 v34; // eax
  _QWORD *v35; // r12
  _BYTE *v36; // r13
  unsigned __int32 v37; // edx
  unsigned __int32 v38; // edi
  _BYTE *v39; // r13
  _QWORD *v40; // r12
  int v41; // ecx
  int v42; // edi
  unsigned int k; // edx
  unsigned int v44; // edx
  int v45; // edi
  __int64 v46; // rcx
  unsigned int j; // edx
  unsigned int v48; // edx
  unsigned int v49; // eax
  int v50; // eax
  unsigned int v51; // ebx
  __int64 v52; // rax
  int v53; // r11d
  _QWORD *v54; // r12
  _BYTE *v55; // r13
  int v56; // r10d
  int v57; // r10d
  const __m128i *v58; // [rsp+10h] [rbp-890h]
  const __m128i *v59; // [rsp+10h] [rbp-890h]
  const __m128i *v60; // [rsp+10h] [rbp-890h]
  const __m128i *v61[2]; // [rsp+40h] [rbp-860h] BYREF
  unsigned __int8 v62; // [rsp+50h] [rbp-850h]
  __m128i v63; // [rsp+58h] [rbp-848h] BYREF
  _QWORD *v64; // [rsp+68h] [rbp-838h] BYREF
  unsigned int v65; // [rsp+70h] [rbp-830h]
  _BYTE v66[56]; // [rsp+868h] [rbp-38h] BYREF

  v4 = (const __m128i *)a1[1].m128i_i64[0];
  v61[0] = a1;
  v62 = 0;
  v61[1] = v4;
  v5 = &v64;
  v63.m128i_i64[0] = 0;
  v63.m128i_i64[1] = 1;
  do
  {
    *(_QWORD *)v5 = 0;
    v5 += 32;
    *((_DWORD *)v5 - 6) = -1;
  }
  while ( v5 != v66 );
  v6 = v61[0][12].m128i_i64[1];
  v7 = *(_QWORD *)((v61[0][12].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 8);
  if ( v6 == v7 )
  {
LABEL_12:
    v16 = 0;
    goto LABEL_13;
  }
  while ( 1 )
  {
    if ( !v6 )
      BUG();
    v8 = *(unsigned __int8 **)(v6 + 32);
    v9 = 16LL * *(unsigned int *)(v6 + 52);
    v10 = &v8[v9];
    if ( v8 != &v8[v9] )
    {
      v11 = 0;
      do
      {
        while ( 1 )
        {
          v13 = *v8;
          if ( !(_BYTE)v13 )
            break;
          v12 = v13 - 14;
          LOBYTE(v12) = (unsigned __int8)v12 <= 0x5Fu;
          v8 += 16;
          v11 |= v12;
          if ( v10 == v8 )
            goto LABEL_10;
        }
        v14 = (__int64)v8;
        v8 += 16;
        LOBYTE(v15) = sub_1F58D20(v14);
        v11 |= v15;
      }
      while ( v10 != v8 );
LABEL_10:
      if ( (_BYTE)v11 )
        break;
    }
    v6 = *(_QWORD *)(v6 + 8);
    if ( v6 == v7 )
      goto LABEL_12;
  }
  sub_1D236A0((__int64)v61[0]->m128i_i64);
  v18 = v61[0];
  v19 = v61[0][12].m128i_i64[1];
  v20 = v61[0][12].m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v19 != *(_QWORD *)(v20 + 8) )
  {
    do
    {
      v21 = v19 - 8;
      if ( !v19 )
        v21 = 0;
      sub_201E5F0((__int64)v61, v21, 0, a2, a3, a4);
      v19 = *(_QWORD *)(v19 + 8);
    }
    while ( v19 != *(_QWORD *)(v20 + 8) );
    v18 = v61[0];
  }
  v22 = v18[11].m128i_u64[0];
  v23 = v18[11].m128i_i32[2];
  v24 = v63.m128i_i8[8] & 1;
  if ( (v63.m128i_i8[8] & 1) != 0 )
  {
    v25 = &v64;
    v26 = 63;
  }
  else
  {
    v26 = v65;
    v25 = v64;
    if ( !v65 )
    {
      v37 = v63.m128i_u32[2];
      ++v63.m128i_i64[0];
      v30 = 0;
      v38 = ((unsigned __int32)v63.m128i_i32[2] >> 1) + 1;
      goto LABEL_44;
    }
    v26 = v65 - 1;
  }
  v27 = 1;
  v28 = 0;
  for ( i = (unsigned int)v26 & (v23 + ((unsigned int)(v22 >> 9) ^ (unsigned int)(v22 >> 4))); ; i = (unsigned int)v26 & v31 )
  {
    v30 = &v25[4 * (unsigned int)i];
    if ( *v30 == v22 && *((_DWORD *)v30 + 2) == v23 )
    {
      v32 = v30[2];
      v33 = v30[3];
      if ( !v32 )
        goto LABEL_50;
      v58 = v18;
      nullsub_686();
      v26 = 0;
      v58[11].m128i_i64[0] = v32;
      v58[11].m128i_i32[2] = v33;
      sub_1D23870();
      goto LABEL_32;
    }
    if ( !*v30 )
      break;
LABEL_26:
    v31 = v27 + i;
    ++v27;
  }
  v53 = *((_DWORD *)v30 + 2);
  if ( v53 != -1 )
  {
    if ( !v28 && v53 == -2 )
      v28 = (__int64)&v25[4 * (unsigned int)i];
    goto LABEL_26;
  }
  v37 = v63.m128i_u32[2];
  v26 = 64;
  if ( v28 )
    v30 = (_QWORD *)v28;
  ++v63.m128i_i64[0];
  v28 = 192;
  v38 = ((unsigned __int32)v63.m128i_i32[2] >> 1) + 1;
  if ( !(_BYTE)v24 )
  {
    v26 = v65;
LABEL_44:
    v28 = (unsigned int)(3 * v26);
  }
  if ( 4 * v38 >= (unsigned int)v28 )
  {
    v60 = v18;
    sub_201A8D0(&v63, 2 * v26);
    v18 = v60;
    if ( (v63.m128i_i8[8] & 1) != 0 )
    {
      v28 = (__int64)&v64;
      v26 = 63;
    }
    else
    {
      v28 = (__int64)v64;
      if ( !v65 )
        goto LABEL_118;
      v26 = v65 - 1;
    }
    v45 = 1;
    v46 = 0;
    for ( j = v26 & (v23 + ((v22 >> 9) ^ (v22 >> 4))); ; j = v26 & v48 )
    {
      v30 = (_QWORD *)(v28 + 32LL * j);
      if ( *v30 == v22 && *((_DWORD *)v30 + 2) == v23 )
        break;
      if ( !*v30 )
      {
        v57 = *((_DWORD *)v30 + 2);
        if ( v57 == -1 )
        {
          if ( !v46 )
            goto LABEL_84;
          v37 = v63.m128i_u32[2];
          v30 = (_QWORD *)v46;
          goto LABEL_47;
        }
        if ( !v46 && v57 == -2 )
          v46 = v28 + 32LL * j;
      }
      v48 = v45 + j;
      ++v45;
    }
    goto LABEL_84;
  }
  if ( (_DWORD)v26 - v63.m128i_i32[3] - v38 <= (unsigned int)v26 >> 3 )
  {
    v59 = v18;
    sub_201A8D0(&v63, v26);
    v18 = v59;
    if ( (v63.m128i_i8[8] & 1) != 0 )
    {
      v28 = (__int64)&v64;
      v41 = 63;
      goto LABEL_63;
    }
    v28 = (__int64)v64;
    if ( v65 )
    {
      v41 = v65 - 1;
LABEL_63:
      v42 = 1;
      v26 = 0;
      for ( k = v41 & (v23 + ((v22 >> 9) ^ (v22 >> 4))); ; k = v41 & v44 )
      {
        v30 = (_QWORD *)(v28 + 32LL * k);
        if ( *v30 == v22 && *((_DWORD *)v30 + 2) == v23 )
          break;
        if ( !*v30 )
        {
          v56 = *((_DWORD *)v30 + 2);
          if ( v56 == -1 )
          {
            if ( v26 )
              v30 = (_QWORD *)v26;
            break;
          }
          if ( !v26 && v56 == -2 )
            v26 = v28 + 32LL * k;
        }
        v44 = v42 + k;
        ++v42;
      }
LABEL_84:
      v37 = v63.m128i_u32[2];
      goto LABEL_47;
    }
LABEL_118:
    v63.m128i_i32[2] = (2 * ((unsigned __int32)v63.m128i_i32[2] >> 1) + 2) | v63.m128i_i8[8] & 1;
    BUG();
  }
LABEL_47:
  v24 = 2 * (v37 >> 1) + 2;
  i = (unsigned int)v24 | v37 & 1;
  v63.m128i_i32[2] = i;
  if ( *v30 || *((_DWORD *)v30 + 2) != -1 )
    --v63.m128i_i32[3];
  *v30 = v22;
  v30[2] = 0;
  *((_DWORD *)v30 + 2) = v23;
  *((_DWORD *)v30 + 6) = 0;
  v33 = v30[3];
LABEL_50:
  v18[11].m128i_i64[0] = 0;
  v18[11].m128i_i32[2] = v33;
LABEL_32:
  ++v63.m128i_i64[0];
  v34 = (unsigned __int32)v63.m128i_i32[2] >> 1;
  if ( (unsigned __int32)v63.m128i_i32[2] >> 1 )
  {
    v24 = 4 * v34;
    if ( (v63.m128i_i8[8] & 1) == 0 )
      goto LABEL_35;
    goto LABEL_42;
  }
  i = v63.m128i_u32[3];
  if ( !v63.m128i_i32[3] )
    goto LABEL_40;
  v24 = 0;
  if ( (v63.m128i_i8[8] & 1) != 0 )
  {
LABEL_42:
    v36 = v66;
    v35 = &v64;
    goto LABEL_38;
  }
LABEL_35:
  i = v65;
  if ( v65 > (unsigned int)v24 && v65 > 0x40 )
  {
    if ( v34 && (v49 = v34 - 1) != 0 )
    {
      _BitScanReverse(&v49, v49);
      v50 = v49 ^ 0x1F;
      v24 = (unsigned int)(33 - v50);
      v51 = 1 << (33 - v50);
      if ( v65 == v51 )
      {
        v63.m128i_i64[1] = v63.m128i_i8[8] & 1;
        if ( v63.m128i_i64[1] )
        {
          v55 = v66;
          v54 = &v64;
        }
        else
        {
          v54 = v64;
          v55 = &v64[4 * v65];
        }
        do
        {
          if ( v54 )
          {
            *v54 = 0;
            *((_DWORD *)v54 + 2) = -1;
          }
          v54 += 4;
        }
        while ( v54 != (_QWORD *)v55 );
        goto LABEL_40;
      }
      j___libc_free_0(v64);
      if ( v51 <= 0x40 )
      {
        v63.m128i_i8[8] |= 1u;
      }
      else
      {
        v63.m128i_i8[8] &= ~1u;
        v52 = sub_22077B0(32LL * v51);
        v65 = v51;
        v64 = (_QWORD *)v52;
      }
    }
    else
    {
      j___libc_free_0(v64);
      v63.m128i_i8[8] |= 1u;
    }
    v63.m128i_i64[1] = v63.m128i_i8[8] & 1;
    if ( v63.m128i_i64[1] )
    {
      v39 = v66;
      v40 = &v64;
    }
    else
    {
      v40 = v64;
      v39 = &v64[4 * v65];
      if ( v64 == (_QWORD *)v39 )
        goto LABEL_40;
    }
    do
    {
      if ( v40 )
      {
        *v40 = 0;
        *((_DWORD *)v40 + 2) = -1;
      }
      v40 += 4;
    }
    while ( v40 != (_QWORD *)v39 );
  }
  else
  {
    v35 = v64;
    v36 = &v64[4 * v65];
    if ( v64 != (_QWORD *)v36 )
    {
      do
      {
LABEL_38:
        *v35 = 0;
        v35 += 4;
        *((_DWORD *)v35 - 6) = -1;
      }
      while ( v35 != (_QWORD *)v36 );
    }
    v63.m128i_i64[1] = v63.m128i_i8[8] & 1;
  }
LABEL_40:
  sub_1D2D9C0(v61[0], v26, i, v24, (__int64)v18, v28);
  v16 = v62;
LABEL_13:
  if ( (v63.m128i_i8[8] & 1) == 0 )
    j___libc_free_0(v64);
  return v16;
}

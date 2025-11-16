// Function: sub_2E91190
// Address: 0x2e91190
//
void __fastcall sub_2E91190(__int64 a1, __int64 a2, const __m128i *a3, __int64 a4, __int64 *a5, __int64 a6)
{
  unsigned __int64 *v6; // rax
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned int v10; // r13d
  __int64 v11; // r14
  __int64 v12; // r12
  unsigned int v13; // r13d
  int v14; // edx
  unsigned int v15; // edi
  __int64 *v16; // rax
  __int64 v17; // r13
  unsigned int v18; // eax
  unsigned __int64 v19; // r14
  unsigned int v20; // r13d
  unsigned int v21; // r12d
  _BYTE *v22; // rdi
  unsigned __int64 v23; // rsi
  unsigned int v24; // edx
  unsigned int v25; // r15d
  __int64 v26; // r14
  __int64 v27; // rcx
  const __m128i *v28; // rax
  __m128i *v29; // r9
  const __m128i *v30; // r13
  const __m128i *v31; // r14
  const __m128i *v32; // rsi
  unsigned __int64 v33; // r14
  _BYTE *v34; // r13
  const __m128i *v35; // rsi
  char v36; // dl
  __int64 *v37; // r13
  __int64 *v38; // r14
  __int64 *v39; // rax
  __int64 v40; // rcx
  __int64 v41; // rcx
  unsigned int v42; // esi
  unsigned int v43; // edx
  unsigned int v44; // esi
  unsigned int v45; // eax
  unsigned int v46; // edx
  unsigned int v47; // edi
  unsigned __int64 v48; // rdx
  const __m128i *v49; // rax
  __m128i *v50; // rdx
  __int64 *v51; // r14
  __int64 v52; // rcx
  int v53; // r11d
  unsigned __int64 v54; // r10
  char *v55; // [rsp+8h] [rbp-108h]
  __int64 v56; // [rsp+10h] [rbp-100h]
  __int64 *v59; // [rsp+38h] [rbp-D8h]
  _OWORD v60[2]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v61; // [rsp+60h] [rbp-B0h]
  __int64 v62; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v63; // [rsp+78h] [rbp-98h]
  __int64 *v64; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v65; // [rsp+88h] [rbp-88h]
  unsigned __int64 v66; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v67; // [rsp+A8h] [rbp-68h]
  _BYTE v68[96]; // [rsp+B0h] [rbp-60h] BYREF

  if ( !a4 )
    return;
  v6 = (unsigned __int64 *)&v64;
  v62 = 0;
  v7 = a1;
  v63 = 1;
  v8 = a4;
  do
    *(_DWORD *)v6++ = -1;
  while ( v6 != &v66 );
  v9 = *(_QWORD *)(a1 + 32);
  v10 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  v11 = v9 + 40LL * v10;
  if ( v11 != v9 )
  {
    v12 = *(_QWORD *)(a1 + 32);
    while ( 1 )
    {
      if ( *(_BYTE *)v12 || (*(_WORD *)(v12 + 2) & 0xFF0) == 0 )
        goto LABEL_7;
      LODWORD(v60[0]) = -858993459 * ((v12 - v9) >> 3);
      v13 = sub_2E89F40(v7, v60[0]);
      if ( (v63 & 1) != 0 )
      {
        a5 = (__int64 *)&v64;
        v14 = 3;
      }
      else
      {
        v44 = v65;
        a5 = v64;
        v14 = v65 - 1;
        if ( !v65 )
        {
          v45 = v63;
          ++v62;
          v66 = 0;
          v46 = ((unsigned int)v63 >> 1) + 1;
          goto LABEL_57;
        }
      }
      v15 = v14 & (37 * LODWORD(v60[0]));
      v16 = &a5[v15];
      a6 = *(unsigned int *)v16;
      if ( LODWORD(v60[0]) != (_DWORD)a6 )
        break;
LABEL_13:
      *((_DWORD *)v16 + 1) = v13;
      v9 = *(_QWORD *)(v7 + 32);
      v17 = v9 + 40LL * LODWORD(v60[0]);
      if ( *(_BYTE *)v17 || (*(_WORD *)(v17 + 2) & 0xFF0) == 0 )
      {
LABEL_7:
        v12 += 40;
        if ( v11 == v12 )
          goto LABEL_16;
      }
      else
      {
        v12 += 40;
        v18 = sub_2E89F40(v7, v60[0]);
        *(_WORD *)(*(_QWORD *)(v7 + 32) + 40LL * v18 + 2) &= 0xF00Fu;
        *(_WORD *)(v17 + 2) &= 0xF00Fu;
        v9 = *(_QWORD *)(v7 + 32);
        if ( v11 == v12 )
        {
LABEL_16:
          v10 = *(_DWORD *)(v7 + 40) & 0xFFFFFF;
          goto LABEL_17;
        }
      }
    }
    v53 = 1;
    v54 = 0;
    while ( (_DWORD)a6 != -1 )
    {
      if ( (_DWORD)a6 == -2 && !v54 )
        v54 = (unsigned __int64)v16;
      v15 = v14 & (v53 + v15);
      v16 = &a5[v15];
      a6 = *(unsigned int *)v16;
      if ( LODWORD(v60[0]) == (_DWORD)a6 )
        goto LABEL_13;
      ++v53;
    }
    v47 = 12;
    v44 = 4;
    if ( !v54 )
      v54 = (unsigned __int64)v16;
    v45 = v63;
    ++v62;
    v66 = v54;
    v46 = ((unsigned int)v63 >> 1) + 1;
    if ( (v63 & 1) == 0 )
    {
      v44 = v65;
LABEL_57:
      v47 = 3 * v44;
    }
    if ( v47 <= 4 * v46 )
    {
      v44 *= 2;
    }
    else if ( v44 - HIDWORD(v63) - v46 > v44 >> 3 )
    {
LABEL_60:
      LODWORD(v63) = (2 * (v45 >> 1) + 2) | v45 & 1;
      v16 = (__int64 *)v66;
      if ( *(_DWORD *)v66 != -1 )
        --HIDWORD(v63);
      *(_QWORD *)v66 = LODWORD(v60[0]);
      goto LABEL_13;
    }
    sub_29758F0((__int64)&v62, v44);
    sub_2AC3C60((__int64)&v62, (int *)v60, &v66);
    v45 = v63;
    goto LABEL_60;
  }
LABEL_17:
  v66 = (unsigned __int64)v68;
  v67 = 0x100000000LL;
  v19 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - v9) >> 3);
  v20 = v10 - v19;
  v21 = -858993459 * ((a2 - v9) >> 3);
  if ( v20 > 1 )
  {
    sub_C8D5F0((__int64)&v66, v68, v20, 0x28u, (__int64)a5, a6);
    v9 = *(_QWORD *)(v7 + 32);
    v24 = v67;
    v23 = HIDWORD(v67);
    v22 = (_BYTE *)v66;
    goto LABEL_20;
  }
  if ( v20 )
  {
    v22 = v68;
    v23 = 1;
    v24 = 0;
LABEL_20:
    v56 = v8;
    v8 = v7;
    v25 = 0;
    v26 = 40LL * (unsigned int)v19;
    while ( 1 )
    {
      v27 = v24;
      v28 = (const __m128i *)(v26 + v9);
      v29 = (__m128i *)&v22[40 * v24];
      if ( v24 >= v23 )
      {
        v48 = v24 + 1LL;
        v60[0] = _mm_loadu_si128(v28);
        v60[1] = _mm_loadu_si128(v28 + 1);
        v61 = v28[2].m128i_i64[0];
        v49 = (const __m128i *)v60;
        if ( v23 < v27 + 1 )
        {
          if ( v22 > (_BYTE *)v60 || v29 <= (__m128i *)v60 )
          {
            sub_C8D5F0((__int64)&v66, v68, v48, 0x28u, (__int64)a5, (__int64)v29);
            v22 = (_BYTE *)v66;
            v27 = (unsigned int)v67;
            v49 = (const __m128i *)v60;
          }
          else
          {
            v55 = (char *)((char *)v60 - v22);
            sub_C8D5F0((__int64)&v66, v68, v48, 0x28u, (__int64)a5, (__int64)v29);
            v22 = (_BYTE *)v66;
            v27 = (unsigned int)v67;
            v49 = (const __m128i *)&v55[v66];
          }
        }
        v50 = (__m128i *)&v22[40 * v27];
        *v50 = _mm_loadu_si128(v49);
        v50[1] = _mm_loadu_si128(v49 + 1);
        v50[2].m128i_i64[0] = v49[2].m128i_i64[0];
        LODWORD(v67) = v67 + 1;
      }
      else
      {
        if ( v29 )
        {
          *v29 = _mm_loadu_si128(v28);
          v29[1] = _mm_loadu_si128(v28 + 1);
          v29[2].m128i_i64[0] = v28[2].m128i_i64[0];
          v24 = v67;
        }
        LODWORD(v67) = v24 + 1;
      }
      ++v25;
      sub_2E8A650(v8, v21);
      if ( v20 <= v25 )
        break;
      v9 = *(_QWORD *)(v8 + 32);
      v24 = v67;
      v23 = HIDWORD(v67);
      v22 = (_BYTE *)v66;
    }
    v7 = v8;
    LODWORD(v8) = v56;
    v30 = (const __m128i *)((char *)a3 + 40 * v56);
    if ( a3 == v30 )
      goto LABEL_30;
    goto LABEL_28;
  }
  v30 = (const __m128i *)((char *)a3 + 40 * v8);
  if ( a3 == v30 )
    goto LABEL_32;
LABEL_28:
  v31 = a3;
  do
  {
    v32 = v31;
    v31 = (const __m128i *)((char *)v31 + 40);
    sub_2E8F270(v7, v32);
  }
  while ( v30 != v31 );
LABEL_30:
  v33 = v66;
  v34 = (_BYTE *)(v66 + 40LL * (unsigned int)v67);
  if ( v34 != (_BYTE *)v66 )
  {
    do
    {
      v35 = (const __m128i *)v33;
      v33 += 40LL;
      sub_2E8F270(v7, v35);
    }
    while ( v34 != (_BYTE *)v33 );
  }
LABEL_32:
  v36 = v63 & 1;
  if ( (unsigned int)v63 >> 1 )
  {
    if ( v36 )
    {
      v37 = (__int64 *)&v66;
      v38 = (__int64 *)&v64;
    }
    else
    {
      v39 = v64;
      v40 = v65;
      v37 = &v64[v65];
      v38 = v64;
      if ( v64 == v37 )
        goto LABEL_39;
    }
    do
    {
      if ( *(_DWORD *)v38 <= 0xFFFFFFFD )
        break;
      ++v38;
    }
    while ( v37 != v38 );
  }
  else
  {
    if ( v36 )
    {
      v51 = (__int64 *)&v64;
      v52 = 4;
    }
    else
    {
      v51 = v64;
      v52 = v65;
    }
    v38 = &v51[v52];
    v37 = v38;
  }
  if ( v36 )
  {
    v39 = (__int64 *)&v64;
    v41 = 4;
    goto LABEL_40;
  }
  v39 = v64;
  v40 = v65;
LABEL_39:
  v41 = v40;
LABEL_40:
  v59 = &v39[v41];
  while ( v38 != v59 )
  {
    v42 = *(_DWORD *)v38;
    v43 = *((_DWORD *)v38 + 1);
    if ( *(_DWORD *)v38 >= v21 )
      v42 = *(_DWORD *)v38 + v8;
    if ( v21 <= v43 )
      v43 += v8;
    sub_2E89ED0(v7, v42, v43);
    do
      ++v38;
    while ( v38 != v37 && *(_DWORD *)v38 > 0xFFFFFFFD );
  }
  if ( (_BYTE *)v66 != v68 )
    _libc_free(v66);
  if ( (v63 & 1) == 0 )
    sub_C7D6A0((__int64)v64, 8LL * v65, 4);
}

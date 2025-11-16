// Function: sub_1F14370
// Address: 0x1f14370
//
__int64 __fastcall sub_1F14370(_QWORD *a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  __int64 v7; // r14
  __int64 v8; // rbx
  int v9; // r15d
  unsigned int v10; // eax
  __int64 v11; // rdx
  unsigned int v12; // r8d
  __int64 v13; // rax
  __int64 *v15; // rbx
  __int64 v16; // r13
  __int64 v17; // r11
  __int64 *v18; // r14
  __int64 v19; // rdx
  __int64 *v20; // rdx
  __int64 v21; // r8
  unsigned __int64 v22; // rax
  __int64 v23; // r15
  int v24; // r9d
  unsigned int v25; // esi
  __int64 v26; // rcx
  __int64 v27; // rcx
  unsigned int v28; // edx
  unsigned int v29; // ecx
  __int64 v30; // r13
  const __m128i *v31; // r10
  __int64 *v32; // r11
  unsigned int v33; // edx
  unsigned int v34; // ecx
  __int64 v35; // rdx
  __m128i v36; // xmm0
  __m128i v37; // xmm1
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // rdx
  __m128i v43; // xmm3
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // rdx
  __m128i v47; // xmm5
  __int64 v48; // rdx
  __int64 v49; // rcx
  unsigned __int64 v50; // rax
  unsigned int v51; // ebx
  int v52; // r15d
  unsigned __int64 v53; // rdx
  unsigned int v54; // r13d
  unsigned __int64 v55; // rdi
  unsigned __int64 v56; // r13
  __int64 v57; // rax
  int v58; // ecx
  unsigned int v59; // r8d
  int v60; // ecx
  __int64 v61; // rdx
  unsigned __int64 v62; // rdx
  unsigned __int64 v63; // rax
  const __m128i *v64; // [rsp+0h] [rbp-B0h]
  int v65; // [rsp+Ch] [rbp-A4h]
  unsigned __int64 v66; // [rsp+10h] [rbp-A0h]
  __int64 v67; // [rsp+18h] [rbp-98h]
  __int64 v68; // [rsp+18h] [rbp-98h]
  __int64 *v69; // [rsp+20h] [rbp-90h]
  const __m128i *v70; // [rsp+20h] [rbp-90h]
  unsigned __int64 v71; // [rsp+28h] [rbp-88h]
  __int64 v72; // [rsp+30h] [rbp-80h]
  __int64 *v73; // [rsp+38h] [rbp-78h]
  __int64 v74; // [rsp+38h] [rbp-78h]
  __int64 v75; // [rsp+38h] [rbp-78h]
  __int64 v76; // [rsp+40h] [rbp-70h]
  __int64 *v77; // [rsp+48h] [rbp-68h]
  unsigned int v78; // [rsp+48h] [rbp-68h]
  __int64 v79; // [rsp+48h] [rbp-68h]
  __int64 v80; // [rsp+50h] [rbp-60h] BYREF
  __int64 v81; // [rsp+58h] [rbp-58h]
  __int64 v82; // [rsp+60h] [rbp-50h]
  __int64 v83; // [rsp+68h] [rbp-48h]
  __int16 v84; // [rsp+70h] [rbp-40h]

  v7 = a1[79];
  v8 = (__int64)(*(_QWORD *)(*a1 + 104LL) - *(_QWORD *)(*a1 + 96LL)) >> 3;
  LOBYTE(v9) = v8;
  if ( (unsigned int)v8 > (unsigned __int64)(v7 << 6) )
  {
    v55 = a1[78];
    v56 = (unsigned int)(v8 + 63) >> 6;
    if ( v56 < 2 * v7 )
      v56 = 2 * v7;
    v57 = (__int64)realloc(v55, 8 * v56, 8 * (int)v56, a4, a5, a6);
    if ( !v57 && (8 * v56 || (v57 = malloc(1u)) == 0) )
    {
      v79 = v57;
      sub_16BD1C0("Allocation failed", 1u);
      v57 = v79;
    }
    v58 = *((_DWORD *)a1 + 160);
    a1[78] = v57;
    a1[79] = v56;
    v59 = (unsigned int)(v58 + 63) >> 6;
    if ( v59 < v56 )
    {
      v78 = (unsigned int)(v58 + 63) >> 6;
      memset((void *)(v57 + 8LL * v59), 0, 8 * (v56 - v59));
      v58 = *((_DWORD *)a1 + 160);
      v59 = v78;
      v57 = a1[78];
    }
    v60 = v58 & 0x3F;
    if ( v60 )
    {
      *(_QWORD *)(v57 + 8LL * (v59 - 1)) &= ~(-1LL << v60);
      v57 = a1[78];
    }
    v61 = a1[79] - (unsigned int)v7;
    if ( v61 )
      memset((void *)(v57 + 8LL * (unsigned int)v7), 0, 8 * v61);
  }
  v10 = *((_DWORD *)a1 + 160);
  if ( (unsigned int)v8 > v10 )
  {
    v53 = a1[79];
    v54 = (v10 + 63) >> 6;
    if ( v53 > v54 )
    {
      v62 = v53 - v54;
      if ( v62 )
      {
        memset((void *)(a1[78] + 8LL * v54), 0, 8 * v62);
        v10 = *((_DWORD *)a1 + 160);
      }
    }
    if ( (v10 & 0x3F) != 0 )
    {
      *(_QWORD *)(a1[78] + 8LL * (v54 - 1)) &= ~(-1LL << (v10 & 0x3F));
      v10 = *((_DWORD *)a1 + 160);
    }
  }
  *((_DWORD *)a1 + 160) = v8;
  if ( (unsigned int)v8 < v10 )
  {
    v50 = a1[79];
    v51 = (unsigned int)(v8 + 63) >> 6;
    if ( v50 > v51 )
    {
      v63 = v50 - v51;
      if ( v63 )
      {
        memset((void *)(a1[78] + 8LL * v51), 0, 8 * v63);
        v9 = *((_DWORD *)a1 + 160);
      }
    }
    v52 = v9 & 0x3F;
    if ( v52 )
      *(_QWORD *)(a1[78] + 8LL * (v51 - 1)) &= ~(-1LL << v52);
  }
  v11 = a1[5];
  v12 = 1;
  *((_DWORD *)a1 + 154) = 0;
  *((_DWORD *)a1 + 162) = 0;
  v13 = *(unsigned int *)(v11 + 8);
  if ( (_DWORD)v13 )
  {
    v15 = *(__int64 **)v11;
    v73 = (__int64 *)a1[25];
    v76 = *(_QWORD *)v11 + 24 * v13;
    v77 = &v73[*((unsigned int *)a1 + 52)];
    v16 = *(_QWORD *)(a1[2] + 272LL);
    v17 = sub_1DA9310(v16, **(_QWORD **)v11);
    v18 = v73;
    while ( 1 )
    {
      v19 = *(unsigned int *)(v17 + 48);
      v81 = 0;
      v82 = 0;
      v20 = (__int64 *)(*(_QWORD *)(v16 + 392) + 16 * v19);
      v83 = 0;
      v21 = v20[1];
      v80 = v17;
      v22 = v21 & 0xFFFFFFFFFFFFFFF8LL;
      v23 = (v21 >> 1) & 3;
      v24 = v23;
      if ( v18 != v77 )
      {
        v25 = v23 | *(_DWORD *)(v22 + 24);
        if ( (*(_DWORD *)((*v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v18 >> 1) & 3) < v25 )
          break;
      }
      ++*((_DWORD *)a1 + 162);
      *(_QWORD *)(a1[78] + 8LL * (*(_DWORD *)(v17 + 48) >> 6)) |= 1LL << *(_DWORD *)(v17 + 48);
      v26 = v15[1];
      if ( (*(_DWORD *)((v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v26 >> 1) & 3) < ((unsigned int)v23
                                                                                            | *(_DWORD *)(v22 + 24)) )
        return 0;
LABEL_10:
      if ( v21 == v26 )
      {
        v15 += 3;
        if ( (__int64 *)v76 == v15 )
          return 1;
      }
      v16 = *(_QWORD *)(a1[2] + 272LL);
      if ( (*(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v15 >> 1) & 3) >= (*(_DWORD *)(v22 + 24)
                                                                                               | (unsigned int)v23) )
        v17 = sub_1DA9310(*(_QWORD *)(a1[2] + 272LL), *v15);
      else
        v17 = *(_QWORD *)(v17 + 8);
    }
    v81 = *v18;
    v27 = *v20;
    do
      ++v18;
    while ( v18 != v77 && v25 > (*(_DWORD *)((*v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v18 >> 1) & 3) );
    v82 = *(v18 - 1);
    v28 = *(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v15 >> 1) & 3;
    v29 = *(_DWORD *)((v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v27 >> 1) & 3;
    LOBYTE(v84) = v28 <= v29;
    if ( v28 > v29 )
      v83 = v81;
    v30 = v15[1];
    HIBYTE(v84) = 1;
    v31 = (const __m128i *)&v80;
    v72 = (__int64)(a1 + 35);
    if ( v25 <= (*(_DWORD *)((v30 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v30 >> 1) & 3) )
    {
LABEL_33:
      v42 = *((unsigned int *)a1 + 72);
      if ( (unsigned int)v42 < *((_DWORD *)a1 + 73) )
        goto LABEL_34;
    }
    else
    {
      v74 = v17;
      v32 = (__int64 *)v76;
      do
      {
        v15 += 3;
        if ( v32 == v15
          || (v33 = *(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v15 >> 1) & 3,
              v34 = v24 | *(_DWORD *)(v22 + 24),
              v33 >= v34) )
        {
          HIBYTE(v84) = 0;
          v17 = v74;
          v82 = v30;
          goto LABEL_33;
        }
        if ( v33 > (*(_DWORD *)((v30 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v30 >> 1) & 3) )
        {
          HIBYTE(v84) = 0;
          v35 = *((unsigned int *)a1 + 72);
          ++*((_DWORD *)a1 + 154);
          if ( (unsigned int)v35 >= *((_DWORD *)a1 + 73) )
          {
            v64 = v31;
            v65 = v24;
            v66 = v22;
            v67 = v21;
            v69 = v32;
            sub_16CD150(v72, a1 + 37, 0, 40, v21, v24);
            v35 = *((unsigned int *)a1 + 72);
            v31 = v64;
            v24 = v65;
            v22 = v66;
            v21 = v67;
            v32 = v69;
          }
          v36 = _mm_loadu_si128(v31);
          v37 = _mm_loadu_si128(v31 + 1);
          v38 = a1[35] + 40 * v35;
          v39 = v31[2].m128i_i64[0];
          v84 = 256;
          *(__m128i *)v38 = v36;
          *(_QWORD *)(v38 + 32) = v39;
          *(__m128i *)(v38 + 16) = v37;
          v40 = a1[35];
          v41 = (unsigned int)(*((_DWORD *)a1 + 72) + 1);
          *((_DWORD *)a1 + 72) = v41;
          *(_QWORD *)(v40 + 40 * v41 - 24) = v30;
          LODWORD(v40) = *(_DWORD *)(v22 + 24);
          v83 = *v15;
          v81 = v83;
          v34 = v24 | v40;
        }
        if ( (v83 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          v83 = *v15;
        v30 = v15[1];
      }
      while ( (*(_DWORD *)((v30 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v30 >> 1) & 3) < v34 );
      v17 = v74;
      v46 = *((unsigned int *)a1 + 72);
      if ( *((_DWORD *)a1 + 73) > (unsigned int)v46 )
      {
        v47 = _mm_loadu_si128(v31 + 1);
        v48 = a1[35] + 40 * v46;
        v49 = v31[2].m128i_i64[0];
        *(__m128i *)v48 = _mm_loadu_si128(v31);
        *(_QWORD *)(v48 + 32) = v49;
        *(__m128i *)(v48 + 16) = v47;
        ++*((_DWORD *)a1 + 72);
LABEL_35:
        v26 = v15[1];
        goto LABEL_10;
      }
    }
    v75 = v21;
    v68 = v17;
    v70 = v31;
    v71 = v22;
    sub_16CD150(v72, a1 + 37, 0, 40, v21, v24);
    v17 = v68;
    v21 = v75;
    v42 = *((unsigned int *)a1 + 72);
    v31 = v70;
    v22 = v71;
LABEL_34:
    v43 = _mm_loadu_si128(v31 + 1);
    v44 = a1[35] + 40 * v42;
    v45 = v31[2].m128i_i64[0];
    *(__m128i *)v44 = _mm_loadu_si128(v31);
    *(_QWORD *)(v44 + 32) = v45;
    *(__m128i *)(v44 + 16) = v43;
    ++*((_DWORD *)a1 + 72);
    if ( (__int64 *)v76 == v15 )
      return 1;
    goto LABEL_35;
  }
  return v12;
}

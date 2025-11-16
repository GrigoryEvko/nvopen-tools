// Function: sub_2A49D10
// Address: 0x2a49d10
//
__int64 *__fastcall sub_2A49D10(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // r8
  _QWORD *v8; // r9
  __int64 *v9; // rbx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rax
  const __m128i *v16; // r15
  char *v17; // rdx
  __int64 v18; // rax
  __m128i *v19; // rax
  __m128i v20; // xmm1
  __int64 v21; // rdx
  int v22; // eax
  unsigned int v23; // eax
  int v24; // r10d
  __int64 v25; // rsi
  __int64 v26; // rdi
  int v27; // r10d
  int v28; // r11d
  unsigned int i; // ecx
  unsigned int v30; // ecx
  __m128i *v31; // r14
  __int64 v32; // rdx
  const __m128i *v33; // r15
  const __m128i *v34; // r15
  __int64 v35; // rbx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 v40; // rdx
  const __m128i *v41; // r14
  __m128i *v42; // rax
  __int64 v43; // rax
  __int64 v44; // r14
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rcx
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rax
  const __m128i *v52; // r15
  char *v53; // rdx
  __int64 v54; // rax
  __m128i *v55; // rax
  __m128i v56; // xmm7
  __int64 v57; // rax
  __int64 v58; // rax
  signed __int64 v59; // r15
  __int64 v60; // rax
  signed __int64 v61; // r14
  signed __int64 v62; // r15
  __int64 v63; // [rsp+10h] [rbp-520h]
  __int64 v64; // [rsp+18h] [rbp-518h]
  __int64 v65; // [rsp+30h] [rbp-500h]
  __int64 *v66; // [rsp+40h] [rbp-4F0h]
  const __m128i *v67; // [rsp+48h] [rbp-4E8h]
  int v68; // [rsp+5Ch] [rbp-4D4h] BYREF
  __m128i **v69; // [rsp+60h] [rbp-4D0h] BYREF
  __int64 v70; // [rsp+68h] [rbp-4C8h]
  __m128i *v71[2]; // [rsp+70h] [rbp-4C0h] BYREF
  __int64 v72; // [rsp+80h] [rbp-4B0h]
  char v73; // [rsp+88h] [rbp-4A8h]
  void *src; // [rsp+1F0h] [rbp-340h] BYREF
  __int64 v75; // [rsp+1F8h] [rbp-338h]
  _BYTE v76[816]; // [rsp+200h] [rbp-330h] BYREF

  v64 = *(_QWORD *)(a1 + 16);
  result = *(__int64 **)a2;
  v63 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 == v63 )
    return result;
  v66 = *(__int64 **)a2;
  do
  {
    v4 = *v66;
    v68 = 0;
    src = v76;
    v65 = v4;
    v75 = 0x1000000000LL;
    v5 = sub_2A453E0(a1, v4);
    v9 = *(__int64 **)v5;
    v10 = *(_QWORD *)v5 + 8LL * *(unsigned int *)(v5 + 8);
    if ( *(_QWORD *)v5 != v10 )
    {
      while ( 1 )
      {
        v69 = 0;
        LODWORD(v70) = 1;
        v71[0] = 0;
        v71[1] = 0;
        v72 = 0;
        v73 = 0;
        v21 = *v9;
        v22 = *(_DWORD *)(*v9 + 24);
        if ( v22 == 1 )
        {
          v6 = *(_QWORD *)(a1 + 16);
          v11 = *(_QWORD *)(*(_QWORD *)(v21 + 56) + 40LL);
          if ( v11 )
          {
            v12 = (unsigned int)(*(_DWORD *)(v11 + 44) + 1);
            v13 = *(_DWORD *)(v11 + 44) + 1;
          }
          else
          {
            v12 = 0;
            v13 = 0;
          }
          if ( *(_DWORD *)(v6 + 32) > v13 )
          {
            v14 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8 * v12);
            if ( v14 )
            {
              v15 = *(_QWORD *)(v14 + 72);
              v6 = HIDWORD(v75);
              v16 = (const __m128i *)&v69;
              v72 = *v9;
              v17 = (char *)src;
              v69 = (__m128i **)v15;
              v18 = (unsigned int)v75;
              v7 = (unsigned int)v75 + 1LL;
              if ( v7 > HIDWORD(v75) )
              {
                if ( src > &v69 || &v69 >= (__m128i ***)((char *)src + 48 * (unsigned int)v75) )
                {
                  sub_C8D5F0((__int64)&src, v76, (unsigned int)v75 + 1LL, 0x30u, v7, (__int64)v8);
                  v17 = (char *)src;
                  v18 = (unsigned int)v75;
                }
                else
                {
                  v59 = (char *)&v69 - (_BYTE *)src;
                  sub_C8D5F0((__int64)&src, v76, (unsigned int)v75 + 1LL, 0x30u, v7, (__int64)v8);
                  v17 = (char *)src;
                  v18 = (unsigned int)v75;
                  v16 = (const __m128i *)((char *)src + v59);
                }
              }
              v19 = (__m128i *)&v17[48 * v18];
              *v19 = _mm_loadu_si128(v16);
              v20 = _mm_loadu_si128(v16 + 1);
              LODWORD(v75) = v75 + 1;
              v19[1] = v20;
              v19[2] = _mm_loadu_si128(v16 + 2);
            }
          }
          goto LABEL_11;
        }
        v23 = v22 & 0xFFFFFFFD;
        if ( !v23 )
          break;
LABEL_11:
        if ( (__int64 *)v10 == ++v9 )
          goto LABEL_20;
      }
      v24 = *(_DWORD *)(a1 + 1640);
      v25 = *(_QWORD *)(v21 + 56);
      v26 = *(_QWORD *)(v21 + 64);
      v7 = *(_QWORD *)(a1 + 1624);
      if ( v24 )
      {
        v27 = v24 - 1;
        v28 = 1;
        for ( i = v27
                & (((0xBF58476D1CE4E5B9LL
                   * (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4)
                    | ((unsigned __int64)(((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4)) << 32))) >> 31)
                 ^ (484763065 * (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4)))); ; i = v27 & v30 )
        {
          v8 = (_QWORD *)(v7 + 16LL * i);
          if ( v25 == *v8 && v26 == v8[1] )
            break;
          if ( *v8 == -4096 && v8[1] == -4096 )
            goto LABEL_50;
          v30 = v28 + i;
          ++v28;
        }
        LODWORD(v70) = 2;
        v6 = *(_QWORD *)(a1 + 16);
        if ( v25 )
        {
          v25 = (unsigned int)(*(_DWORD *)(v25 + 44) + 1);
          v23 = v25;
        }
        if ( v23 >= *(_DWORD *)(v6 + 32) )
          goto LABEL_11;
        v57 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8 * v25);
        if ( !v57 )
          goto LABEL_11;
        v58 = *(_QWORD *)(v57 + 72);
        v6 = HIDWORD(v75);
        v52 = (const __m128i *)&v69;
        v72 = v21;
        v73 = 1;
        v53 = (char *)src;
        v69 = (__m128i **)v58;
        v54 = (unsigned int)v75;
        v7 = (unsigned int)v75 + 1LL;
        if ( v7 <= HIDWORD(v75) )
          goto LABEL_55;
        if ( src > &v69 || &v69 >= (__m128i ***)((char *)src + 48 * (unsigned int)v75) )
          goto LABEL_78;
      }
      else
      {
LABEL_50:
        LODWORD(v70) = 0;
        v6 = *(_QWORD *)(a1 + 16);
        v49 = 0;
        if ( v26 )
        {
          v49 = (unsigned int)(*(_DWORD *)(v26 + 44) + 1);
          v23 = *(_DWORD *)(v26 + 44) + 1;
        }
        if ( v23 >= *(_DWORD *)(v6 + 32) )
          goto LABEL_11;
        v50 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8 * v49);
        if ( !v50 )
          goto LABEL_11;
        v51 = *(_QWORD *)(v50 + 72);
        v6 = HIDWORD(v75);
        v52 = (const __m128i *)&v69;
        v72 = v21;
        v53 = (char *)src;
        v69 = (__m128i **)v51;
        v54 = (unsigned int)v75;
        v7 = (unsigned int)v75 + 1LL;
        if ( v7 <= HIDWORD(v75) )
          goto LABEL_55;
        if ( src > &v69 || &v69 >= (__m128i ***)((char *)src + 48 * (unsigned int)v75) )
        {
LABEL_78:
          v52 = (const __m128i *)&v69;
          sub_C8D5F0((__int64)&src, v76, (unsigned int)v75 + 1LL, 0x30u, v7, (__int64)v8);
          v53 = (char *)src;
          v54 = (unsigned int)v75;
LABEL_55:
          v55 = (__m128i *)&v53[48 * v54];
          *v55 = _mm_loadu_si128(v52);
          v56 = _mm_loadu_si128(v52 + 1);
          LODWORD(v75) = v75 + 1;
          v55[1] = v56;
          v55[2] = _mm_loadu_si128(v52 + 2);
          goto LABEL_11;
        }
      }
      v62 = (char *)&v69 - (_BYTE *)src;
      sub_C8D5F0((__int64)&src, v76, (unsigned int)v75 + 1LL, 0x30u, v7, (__int64)v8);
      v53 = (char *)src;
      v54 = (unsigned int)v75;
      v52 = (const __m128i *)((char *)src + v62);
      goto LABEL_55;
    }
LABEL_20:
    sub_2A45170(a1, v65, (__int64)&src, v6, v7, (__int64)v8);
    v31 = (__m128i *)src;
    v32 = 48LL * (unsigned int)v75;
    v33 = (const __m128i *)((char *)src + v32);
    sub_2A45B30((__int64 *)&v69, (__m128i *)src, 0xAAAAAAAAAAAAAAABLL * (v32 >> 4));
    if ( v71[0] )
      sub_2A49BF0(v31, v33, v71[0], v70, v64);
    else
      sub_2A47830(v31, v33, v64);
    j_j___libc_free_0((unsigned __int64)v71[0]);
    v34 = (const __m128i *)src;
    v69 = v71;
    v70 = 0x800000000LL;
    if ( (char *)src + 48 * (unsigned int)v75 == src )
      goto LABEL_42;
    v67 = (const __m128i *)((char *)src + 48 * (unsigned int)v75);
    do
    {
      while ( 1 )
      {
        v35 = v34[2].m128i_i64[0];
        if ( v34[1].m128i_i64[0] | v35 )
        {
          sub_2A45070(a1, (__int64)&v69, (__int64)v34);
          sub_2A45120(a1, (__int64)&v69, (__int64)v34);
          v38 = (unsigned int)v70;
          v39 = (unsigned int)v70 + 1LL;
          if ( v39 > HIDWORD(v70) )
          {
            if ( v69 > (__m128i **)v34 || v34 >= (const __m128i *)&v69[6 * (unsigned int)v70] )
            {
              v41 = v34;
              sub_C8D5F0((__int64)&v69, v71, v39, 0x30u, v36, v37);
              v40 = (__int64)v69;
              v38 = (unsigned int)v70;
            }
            else
            {
              v61 = (char *)v34 - (char *)v69;
              sub_C8D5F0((__int64)&v69, v71, v39, 0x30u, v36, v37);
              v40 = (__int64)v69;
              v38 = (unsigned int)v70;
              v41 = (const __m128i *)((char *)v69 + v61);
            }
          }
          else
          {
            v40 = (__int64)v69;
            v41 = v34;
          }
          v42 = (__m128i *)(v40 + 48 * v38);
          *v42 = _mm_loadu_si128(v41);
          v42[1] = _mm_loadu_si128(v41 + 1);
          v42[2] = _mm_loadu_si128(v41 + 2);
          v43 = (unsigned int)(v70 + 1);
          LODWORD(v70) = v70 + 1;
        }
        else
        {
          if ( !(unsigned __int8)sub_2A45070(a1, (__int64)&v69, (__int64)v34) )
            sub_2A45120(a1, (__int64)&v69, (__int64)v34);
          v43 = (unsigned int)v70;
        }
        if ( !(_DWORD)v43 || v34[1].m128i_i64[0] | v35 )
          goto LABEL_24;
        v44 = (__int64)&v69[6 * v43 - 6];
        v45 = *(_QWORD *)(v44 + 16);
        if ( v45 )
          break;
        v60 = sub_2A465E0((_QWORD *)a1, &v68, (__int64 *)&v69, v65);
        *(_QWORD *)(v44 + 16) = v60;
        v45 = v60;
        v46 = v34[1].m128i_i64[1];
        if ( *(_QWORD *)v46 )
        {
          v47 = *(_QWORD *)(v46 + 8);
          **(_QWORD **)(v46 + 16) = v47;
          if ( v47 )
LABEL_34:
            *(_QWORD *)(v47 + 16) = *(_QWORD *)(v46 + 16);
        }
        *(_QWORD *)v46 = v45;
        if ( v45 )
          goto LABEL_36;
LABEL_24:
        v34 += 3;
        if ( v34 == v67 )
          goto LABEL_39;
      }
      v46 = v34[1].m128i_i64[1];
      if ( *(_QWORD *)v46 )
      {
        v47 = *(_QWORD *)(v46 + 8);
        **(_QWORD **)(v46 + 16) = v47;
        if ( v47 )
          goto LABEL_34;
      }
      *(_QWORD *)v46 = v45;
LABEL_36:
      v48 = *(_QWORD *)(v45 + 16);
      *(_QWORD *)(v46 + 8) = v48;
      if ( v48 )
        *(_QWORD *)(v48 + 16) = v46 + 8;
      *(_QWORD *)(v46 + 16) = v45 + 16;
      v34 += 3;
      *(_QWORD *)(v45 + 16) = v46;
    }
    while ( v34 != v67 );
LABEL_39:
    if ( v69 != v71 )
      _libc_free((unsigned __int64)v69);
    v34 = (const __m128i *)src;
LABEL_42:
    if ( v34 != (const __m128i *)v76 )
      _libc_free((unsigned __int64)v34);
    result = ++v66;
  }
  while ( (__int64 *)v63 != v66 );
  return result;
}

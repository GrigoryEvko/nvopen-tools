// Function: sub_3546200
// Address: 0x3546200
//
__int64 __fastcall sub_3546200(__int64 a1, const __m128i *a2, _QWORD *a3)
{
  unsigned int v3; // ecx
  unsigned __int64 *v5; // rdi
  unsigned int v6; // r12d
  __m128i v7; // xmm2
  __m128i v8; // xmm3
  char v9; // dl
  __int64 v10; // rax
  unsigned __int64 v11; // r14
  unsigned __int64 *v12; // rax
  unsigned __int64 *v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 *v19; // rdx
  __int64 v20; // rcx
  unsigned __int64 v21; // r8
  unsigned __int64 v22; // r9
  __int64 v23; // r10
  __int64 v24; // rax
  __int64 v25; // rax
  const __m128i *v26; // r11
  __m128i *v27; // rax
  __m128i v28; // xmm1
  unsigned __int64 *v29; // rax
  __int64 v31; // [rsp+8h] [rbp-1C8h]
  const __m128i *v32; // [rsp+8h] [rbp-1C8h]
  unsigned __int64 v33; // [rsp+10h] [rbp-1C0h]
  __int64 v34; // [rsp+10h] [rbp-1C0h]
  const __m128i *v35; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v36; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v37; // [rsp+18h] [rbp-1B8h]
  __int64 v39; // [rsp+30h] [rbp-1A0h] BYREF
  unsigned __int64 *v40; // [rsp+38h] [rbp-198h]
  __int64 v41; // [rsp+40h] [rbp-190h]
  int v42; // [rsp+48h] [rbp-188h]
  char v43; // [rsp+4Ch] [rbp-184h]
  char v44; // [rsp+50h] [rbp-180h] BYREF
  unsigned __int64 *v45; // [rsp+90h] [rbp-140h] BYREF
  __int64 v46; // [rsp+98h] [rbp-138h]
  _OWORD v47[19]; // [rsp+A0h] [rbp-130h] BYREF

  v3 = 1;
  v5 = (unsigned __int64 *)v47;
  v6 = 0x80000000;
  v7 = _mm_loadu_si128(a2);
  v8 = _mm_loadu_si128(a2 + 1);
  v40 = (unsigned __int64 *)&v44;
  v39 = 0;
  v46 = 0x800000001LL;
  v41 = 8;
  v42 = 0;
  v43 = 1;
  v45 = (unsigned __int64 *)v47;
  v9 = 1;
  v47[0] = v7;
  v47[1] = v8;
  while ( 1 )
  {
    v10 = v3--;
    v11 = v5[4 * v10 - 4];
    LODWORD(v46) = v3;
    if ( !v9 )
    {
      if ( sub_C8CA60((__int64)&v39, v11) )
        goto LABEL_32;
      goto LABEL_10;
    }
    v12 = v40;
    v13 = &v40[HIDWORD(v41)];
    if ( v40 != v13 )
      break;
LABEL_10:
    if ( *(_DWORD *)(v11 + 200) == -1 )
      goto LABEL_32;
    v14 = *(_QWORD **)(a1 + 48);
    v15 = (_QWORD *)(a1 + 40);
    if ( !v14 )
      goto LABEL_32;
    do
    {
      while ( 1 )
      {
        v16 = v14[2];
        v17 = v14[3];
        if ( v14[4] >= v11 )
          break;
        v14 = (_QWORD *)v14[3];
        if ( !v17 )
          goto LABEL_16;
      }
      v15 = v14;
      v14 = (_QWORD *)v14[2];
    }
    while ( v16 );
LABEL_16:
    if ( (_QWORD *)(a1 + 40) == v15 || v15[4] > v11 )
      goto LABEL_32;
    if ( (signed int)v6 < *((_DWORD *)v15 + 10) )
      v6 = *((_DWORD *)v15 + 10);
    v18 = sub_3545E90(a3, v11);
    v22 = *(_QWORD *)v18;
    v23 = *(_QWORD *)v18 + 32LL * *(unsigned int *)(v18 + 8);
    if ( v23 != *(_QWORD *)v18 )
    {
      do
      {
        while ( 1 )
        {
          v24 = (*(__int64 *)(v22 + 8) >> 1) & 3;
          if ( v24 == 2 || v24 == 3 )
            break;
          v22 += 32LL;
          if ( v22 == v23 )
            goto LABEL_27;
        }
        v25 = (unsigned int)v46;
        v20 = HIDWORD(v46);
        v26 = (const __m128i *)v22;
        v19 = v45;
        v21 = (unsigned int)v46 + 1LL;
        if ( v21 > HIDWORD(v46) )
        {
          if ( (unsigned __int64)v45 > v22 )
          {
            v32 = (const __m128i *)v22;
            v34 = v23;
            v37 = v22;
            sub_C8D5F0((__int64)&v45, v47, (unsigned int)v46 + 1LL, 0x20u, v21, v22);
            v19 = v45;
            v25 = (unsigned int)v46;
            v22 = v37;
            v23 = v34;
            v26 = v32;
          }
          else
          {
            v35 = (const __m128i *)v22;
            v31 = v23;
            v33 = v22;
            if ( v22 >= (unsigned __int64)&v45[4 * (unsigned int)v46] )
            {
              sub_C8D5F0((__int64)&v45, v47, (unsigned int)v46 + 1LL, 0x20u, v21, v22);
              v19 = v45;
              v25 = (unsigned int)v46;
              v22 = v33;
              v23 = v31;
              v26 = v35;
            }
            else
            {
              v36 = v22 - (_QWORD)v45;
              sub_C8D5F0((__int64)&v45, v47, (unsigned int)v46 + 1LL, 0x20u, v21, v22);
              v19 = v45;
              v23 = v31;
              v22 = v33;
              v26 = (const __m128i *)((char *)v45 + v36);
              v25 = (unsigned int)v46;
            }
          }
        }
        v22 += 32LL;
        v27 = (__m128i *)&v19[4 * v25];
        *v27 = _mm_loadu_si128(v26);
        v28 = _mm_loadu_si128(v26 + 1);
        LODWORD(v46) = v46 + 1;
        v27[1] = v28;
      }
      while ( v22 != v23 );
    }
LABEL_27:
    if ( !v43 )
      goto LABEL_40;
    v29 = v40;
    v20 = HIDWORD(v41);
    v19 = &v40[HIDWORD(v41)];
    if ( v40 != v19 )
    {
      while ( v11 != *v29 )
      {
        if ( v19 == ++v29 )
          goto LABEL_39;
      }
LABEL_32:
      v3 = v46;
      v5 = v45;
      goto LABEL_33;
    }
LABEL_39:
    if ( HIDWORD(v41) < (unsigned int)v41 )
    {
      ++HIDWORD(v41);
      *v19 = v11;
      v3 = v46;
      ++v39;
      v5 = v45;
    }
    else
    {
LABEL_40:
      sub_C8CC70((__int64)&v39, v11, (__int64)v19, v20, v21, v22);
      v3 = v46;
      v5 = v45;
    }
LABEL_33:
    if ( !v3 )
      goto LABEL_34;
LABEL_8:
    v9 = v43;
  }
  while ( v11 != *v12 )
  {
    if ( v13 == ++v12 )
      goto LABEL_10;
  }
  if ( v3 )
    goto LABEL_8;
LABEL_34:
  if ( v5 != (unsigned __int64 *)v47 )
    _libc_free((unsigned __int64)v5);
  if ( !v43 )
    _libc_free((unsigned __int64)v40);
  return v6;
}

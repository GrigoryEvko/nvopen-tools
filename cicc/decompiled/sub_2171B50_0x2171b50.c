// Function: sub_2171B50
// Address: 0x2171b50
//
void __fastcall sub_2171B50(__int64 a1, __int64 *a2, __int64 a3, __m128i a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // rsi
  int v9; // eax
  __int64 v10; // rdx
  const __m128i *v11; // r10
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  int v14; // eax
  __m128i v15; // xmm1
  __m128i v16; // xmm4
  __m128i v17; // xmm5
  __int64 v18; // rbx
  unsigned int v19; // ecx
  const __m128i *v20; // r12
  __int64 i; // rax
  unsigned int v22; // edx
  __int64 v23; // rax
  __int64 *v24; // rax
  unsigned int *v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // r13
  __int64 v28; // rdx
  char v29; // si
  const void **v30; // rdx
  unsigned __int8 *v31; // rax
  char v32; // di
  char v33; // al
  __int64 v34; // rax
  unsigned __int8 *v35; // rax
  unsigned __int8 *v36; // rsi
  const void ***v37; // rax
  int v38; // edx
  __int64 v39; // r9
  int v40; // r8d
  int v41; // r9d
  __int64 *v42; // r12
  __int64 v43; // rax
  int v44; // ebx
  __int64 *v45; // r13
  __int64 **v46; // rax
  __int128 v47; // [rsp-10h] [rbp-180h]
  __int64 v48; // [rsp+8h] [rbp-168h]
  __int64 v49; // [rsp+10h] [rbp-160h]
  int v51; // [rsp+34h] [rbp-13Ch]
  __int64 m128i_i64; // [rsp+40h] [rbp-130h]
  __int64 v54; // [rsp+50h] [rbp-120h] BYREF
  int v55; // [rsp+58h] [rbp-118h]
  __int64 v56; // [rsp+60h] [rbp-110h] BYREF
  const void **v57; // [rsp+68h] [rbp-108h]
  unsigned __int8 *v58; // [rsp+70h] [rbp-100h] BYREF
  __int64 v59; // [rsp+78h] [rbp-F8h]
  _BYTE v60[48]; // [rsp+80h] [rbp-F0h] BYREF
  _OWORD *v61; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v62; // [rsp+B8h] [rbp-B8h]
  _OWORD v63[11]; // [rsp+C0h] [rbp-B0h] BYREF

  v8 = *(_QWORD *)(a1 + 72);
  v54 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v54, v8, 2);
  v62 = 0x800000000LL;
  v9 = *(_DWORD *)(a1 + 64);
  v10 = *(_QWORD *)(a1 + 32);
  v61 = v63;
  v11 = *(const __m128i **)(a1 + 40);
  v12 = _mm_loadu_si128((const __m128i *)(v10 + 40));
  v13 = _mm_loadu_si128((const __m128i *)(v10 + 80));
  v55 = v9;
  v14 = *(_DWORD *)(a1 + 60);
  v15 = _mm_loadu_si128((const __m128i *)v10);
  LODWORD(v62) = 5;
  v16 = _mm_loadu_si128((const __m128i *)(v10 + 120));
  v17 = _mm_loadu_si128((const __m128i *)(v10 + 160));
  v58 = v60;
  v51 = v14 - 1;
  v59 = 0x300000000LL;
  v63[0] = v15;
  v63[1] = v12;
  v63[2] = v13;
  v63[3] = v16;
  v63[4] = v17;
  if ( v14 - 1 <= 0 )
  {
    v34 = 0;
  }
  else
  {
    v18 = 200;
    v19 = 3;
    v20 = v11;
    m128i_i64 = (__int64)v11[(unsigned int)(v14 - 2) + 1].m128i_i64;
    for ( i = 0; ; i = (unsigned int)v59 )
    {
      v25 = (unsigned int *)(v18 + v10);
      v26 = *(_QWORD *)(*(_QWORD *)v25 + 88LL);
      v27 = *(_QWORD *)(v26 + 24);
      if ( *(_DWORD *)(v26 + 32) > 0x40u )
        v27 = **(_QWORD **)(v26 + 24);
      v28 = *(_QWORD *)(*(_QWORD *)v25 + 40LL) + 16LL * v25[2];
      v29 = *(_BYTE *)v28;
      v30 = *(const void ***)(v28 + 8);
      LOBYTE(v56) = v29;
      v57 = v30;
      if ( v20->m128i_i8[0] == 3 )
      {
        if ( (unsigned int)i >= v19 )
        {
          sub_16CD150((__int64)&v58, v60, 0, 16, a6, a7);
          i = (unsigned int)v59;
        }
        v31 = &v58[16 * i];
        *(_QWORD *)v31 = 4;
        v32 = v56;
        *((_QWORD *)v31 + 1) = 0;
        LODWORD(v59) = v59 + 1;
        v33 = v32 ? sub_216FFF0(v32) : sub_1F58D40((__int64)&v56);
        v27 |= 1LL << (v33 - 1);
      }
      else
      {
        if ( (unsigned int)i >= v19 )
        {
          sub_16CD150((__int64)&v58, v60, 0, 16, a6, a7);
          i = (unsigned int)v59;
        }
        a4 = _mm_loadu_si128(v20);
        *(__m128i *)&v58[16 * i] = a4;
        LODWORD(v59) = v59 + 1;
      }
      a6 = sub_1D38BB0((__int64)a2, v27, (__int64)&v54, v56, v57, 1, a4, *(double *)v15.m128i_i64, v12, 0);
      a7 = v22;
      v23 = (unsigned int)v62;
      if ( (unsigned int)v62 >= HIDWORD(v62) )
      {
        v48 = a6;
        v49 = v22;
        sub_16CD150((__int64)&v61, v63, 0, 16, a6, v22);
        v23 = (unsigned int)v62;
        a6 = v48;
        a7 = v49;
      }
      v24 = (__int64 *)&v61[v23];
      v18 += 40;
      ++v20;
      *v24 = a6;
      v24[1] = a7;
      LODWORD(v62) = v62 + 1;
      if ( (const __m128i *)m128i_i64 == v20 )
        break;
      v19 = HIDWORD(v59);
      v10 = *(_QWORD *)(a1 + 32);
    }
    v34 = (unsigned int)v59;
    if ( HIDWORD(v59) <= (unsigned int)v59 )
    {
      sub_16CD150((__int64)&v58, v60, 0, 16, a6, a7);
      v34 = (unsigned int)v59;
    }
  }
  v35 = &v58[16 * v34];
  *(_QWORD *)v35 = 1;
  v36 = v58;
  *((_QWORD *)v35 + 1) = 0;
  LODWORD(v59) = v59 + 1;
  v37 = (const void ***)sub_1D25C30((__int64)a2, v36, (unsigned int)v59);
  *((_QWORD *)&v47 + 1) = (unsigned int)v62;
  *(_QWORD *)&v47 = v61;
  v42 = sub_1D36D80(a2, 204, (__int64)&v54, v37, v38, *(double *)a4.m128i_i64, *(double *)v15.m128i_i64, v12, v39, v47);
  if ( v51 >= 0 )
  {
    v43 = *(unsigned int *)(a3 + 8);
    v44 = 0;
    do
    {
      v45 = (__int64 *)(unsigned int)v44;
      if ( (unsigned int)v43 >= *(_DWORD *)(a3 + 12) )
      {
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v40, v41);
        v43 = *(unsigned int *)(a3 + 8);
      }
      v46 = (__int64 **)(*(_QWORD *)a3 + 16 * v43);
      ++v44;
      *v46 = v42;
      v46[1] = v45;
      v43 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
      *(_DWORD *)(a3 + 8) = v43;
    }
    while ( v51 >= v44 );
  }
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  if ( v61 != v63 )
    _libc_free((unsigned __int64)v61);
  if ( v54 )
    sub_161E7C0((__int64)&v54, v54);
}

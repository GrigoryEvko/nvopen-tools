// Function: sub_21BF200
// Address: 0x21bf200
//
void __fastcall sub_21BF200(
        __int64 a1,
        __int64 a2,
        double a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // rax
  __int64 v12; // rsi
  __m128i v13; // xmm1
  void *v14; // rcx
  int v15; // r15d
  __int64 v16; // r8
  int v17; // r13d
  int v18; // eax
  _QWORD *v19; // rdi
  __int64 v20; // rax
  __int64 v21; // r9
  unsigned __int64 v22; // r8
  const void *v23; // r11
  size_t v24; // r15
  int v25; // eax
  unsigned __int64 v26; // rdx
  int v27; // r8d
  __int64 *v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // rdx
  __m128i v31; // xmm0
  __int64 v32; // rcx
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  __int64 *v35; // r13
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  unsigned __int8 *v40; // rdi
  __int128 v41; // [rsp-10h] [rbp-1B0h]
  __int128 v42; // [rsp-10h] [rbp-1B0h]
  int v43; // [rsp+0h] [rbp-1A0h]
  const void *v44; // [rsp+0h] [rbp-1A0h]
  int v45; // [rsp+8h] [rbp-198h]
  unsigned __int64 v46; // [rsp+8h] [rbp-198h]
  int v47; // [rsp+8h] [rbp-198h]
  __m128i v48; // [rsp+10h] [rbp-190h] BYREF
  __int64 v49; // [rsp+20h] [rbp-180h]
  void *dest; // [rsp+28h] [rbp-178h]
  __m128i v51; // [rsp+30h] [rbp-170h]
  __int64 v52; // [rsp+40h] [rbp-160h] BYREF
  int v53; // [rsp+48h] [rbp-158h]
  unsigned __int8 *v54; // [rsp+50h] [rbp-150h] BYREF
  __int64 v55; // [rsp+58h] [rbp-148h]
  _BYTE v56[128]; // [rsp+60h] [rbp-140h] BYREF
  __int64 *v57; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v58; // [rsp+E8h] [rbp-B8h]
  __int64 v59; // [rsp+F0h] [rbp-B0h] BYREF
  int v60; // [rsp+F8h] [rbp-A8h]

  v11 = *(_QWORD *)(a2 + 32);
  v12 = *(_QWORD *)(a2 + 72);
  v13 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v14 = *(void **)(v11 + 80);
  v52 = v12;
  v15 = *(_DWORD *)(v11 + 88);
  v16 = *(_QWORD *)(v11 + 120);
  v17 = *(_DWORD *)(v11 + 128);
  v48 = v13;
  if ( v12 )
  {
    v49 = v16;
    dest = v14;
    sub_1623A60((__int64)&v52, v12, 2);
    v16 = v49;
    v14 = dest;
  }
  v60 = v17;
  v18 = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v41 + 1) = 2;
  v19 = *(_QWORD **)(a1 + 272);
  *(_QWORD *)&v41 = &v57;
  v57 = (__int64 *)v14;
  v59 = v16;
  LODWORD(v58) = v15;
  v53 = v18;
  v20 = sub_1D2CDB0(v19, 4451, (__int64)&v52, 7, 0, a9, v41);
  v22 = *(unsigned int *)(a2 + 60);
  v23 = *(const void **)(a2 + 40);
  v49 = v20;
  dest = v56;
  v54 = v56;
  v24 = 16 * v22;
  v55 = 0x800000000LL;
  if ( v22 > 8 )
  {
    v44 = v23;
    v47 = v22;
    sub_16CD150((__int64)&v54, dest, v22, 16, v22, v21);
    LODWORD(v22) = v47;
    v23 = v44;
    v40 = &v54[16 * (unsigned int)v55];
  }
  else
  {
    if ( !v24 )
      goto LABEL_5;
    v40 = (unsigned __int8 *)dest;
  }
  v45 = v22;
  memcpy(v40, v23, v24);
  LODWORD(v24) = v55;
  LODWORD(v22) = v45;
LABEL_5:
  v25 = *(_DWORD *)(a2 + 56);
  LODWORD(v55) = v24 + v22;
  v26 = (unsigned int)(v25 - 1);
  v57 = &v59;
  v58 = 0x800000000LL;
  v27 = v26;
  v28 = &v59;
  if ( (unsigned int)v26 > 8 )
  {
    v43 = v26;
    v46 = v26;
    sub_16CD150((__int64)&v57, &v59, v26, 16, v26, v21);
    v28 = v57;
    v27 = v43;
    v26 = v46;
  }
  LODWORD(v58) = v27;
  v29 = &v28[2 * v26];
  if ( v29 != v28 )
  {
    do
    {
      if ( v28 )
      {
        *v28 = 0;
        *((_DWORD *)v28 + 2) = 0;
      }
      v28 += 2;
    }
    while ( v29 != v28 );
    v28 = v57;
  }
  v30 = *(_QWORD *)(a2 + 32);
  v31 = _mm_load_si128(&v48);
  *v28 = *(_QWORD *)v30;
  v32 = v49;
  *((_DWORD *)v28 + 2) = *(_DWORD *)(v30 + 8);
  v33 = (unsigned __int64)v57;
  v51 = v31;
  v57[2] = v31.m128i_i64[0];
  LODWORD(v30) = v51.m128i_i32[2];
  *(_QWORD *)(v33 + 32) = v32;
  *(_DWORD *)(v33 + 24) = v30;
  *(_DWORD *)(v33 + 40) = 0;
  if ( *(_DWORD *)(a2 + 56) == 5 )
  {
    v34 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(v33 + 48) = *(_QWORD *)(v34 + 160);
    *(_DWORD *)(v33 + 56) = *(_DWORD *)(v34 + 168);
  }
  *((_QWORD *)&v42 + 1) = (unsigned int)v58;
  *(_QWORD *)&v42 = v33;
  v35 = sub_1D373B0(
          *(__int64 **)(a1 + 272),
          0x2Eu,
          (__int64)&v52,
          v54,
          (unsigned int)v55,
          *(double *)v31.m128i_i64,
          *(double *)v13.m128i_i64,
          a5,
          v21,
          v42);
  sub_1D444E0(*(_QWORD *)(a1 + 272), a2, (__int64)v35);
  sub_1D49010((__int64)v35);
  sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v36, v37, v38, v39);
  if ( v57 != &v59 )
    _libc_free((unsigned __int64)v57);
  if ( v54 != dest )
    _libc_free((unsigned __int64)v54);
  if ( v52 )
    sub_161E7C0((__int64)&v52, v52);
}

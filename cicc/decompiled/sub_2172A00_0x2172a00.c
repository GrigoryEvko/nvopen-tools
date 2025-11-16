// Function: sub_2172A00
// Address: 0x2172a00
//
void __fastcall sub_2172A00(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  __m128i v10; // xmm2
  __m128i v11; // xmm1
  __m128i v12; // xmm0
  char *v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  char v17; // cl
  __int64 v18; // rdx
  __int64 v19; // rdx
  char v20; // cl
  __int64 v21; // rdx
  __int64 *v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rdi
  unsigned __int8 *v25; // rcx
  __int64 v26; // r8
  __int64 *v27; // r13
  __int64 *v28; // rdx
  __int64 *v29; // r8
  __int64 v30; // rax
  __int64 *v31; // r9
  __int64 **v32; // rax
  __int64 v33; // rax
  __int64 **v34; // rax
  __int64 v35; // rax
  __int64 **v36; // rax
  __int64 *v37; // rdi
  __int128 v38; // [rsp-20h] [rbp-1F0h]
  __int128 v39; // [rsp-10h] [rbp-1E0h]
  __int64 *v40; // [rsp+10h] [rbp-1C0h]
  __int64 *v41; // [rsp+18h] [rbp-1B8h]
  __int64 v42; // [rsp+50h] [rbp-180h] BYREF
  int v43; // [rsp+58h] [rbp-178h]
  __int64 *v44; // [rsp+60h] [rbp-170h] BYREF
  int v45; // [rsp+68h] [rbp-168h]
  __int64 *v46; // [rsp+70h] [rbp-160h]
  int v47; // [rsp+78h] [rbp-158h]
  unsigned __int8 *v48; // [rsp+80h] [rbp-150h]
  __int64 v49; // [rsp+88h] [rbp-148h]
  _BYTE v50[64]; // [rsp+90h] [rbp-140h] BYREF
  char v51; // [rsp+D0h] [rbp-100h] BYREF
  __int64 *v52; // [rsp+110h] [rbp-C0h]
  __int64 v53; // [rsp+118h] [rbp-B8h]
  __int64 v54; // [rsp+120h] [rbp-B0h] BYREF
  int v55; // [rsp+128h] [rbp-A8h]
  __int64 v56; // [rsp+130h] [rbp-A0h]
  int v57; // [rsp+138h] [rbp-98h]
  __int64 v58; // [rsp+140h] [rbp-90h]
  int v59; // [rsp+148h] [rbp-88h]

  v8 = *(_QWORD *)(a1 + 72);
  v42 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v42, v8, 2);
  v43 = *(_DWORD *)(a1 + 64);
  v9 = *(_QWORD *)(a1 + 32);
  v10 = _mm_loadu_si128((const __m128i *)v9);
  v48 = v50;
  v11 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v12 = _mm_loadu_si128((const __m128i *)(v9 + 80));
  v49 = 0x800000004LL;
  v13 = v50;
  do
  {
    *v13 = 0;
    v13 += 16;
    *((_QWORD *)v13 - 1) = 0;
  }
  while ( v13 != &v51 );
  v59 = 0;
  v53 = 0x800000003LL;
  v14 = (unsigned __int64)v48;
  v58 = 0;
  v57 = 0;
  v56 = 0;
  v55 = 0;
  v52 = &v54;
  v54 = 0;
  *v48 = 6;
  *(_QWORD *)(v14 + 8) = 0;
  v15 = (unsigned __int64)v48;
  v48[16] = 6;
  *(_QWORD *)(v15 + 24) = 0;
  v16 = *(_QWORD *)(a1 + 40);
  v17 = *(_BYTE *)(v16 + 16);
  v18 = *(_QWORD *)(v16 + 24);
  *(_BYTE *)(v15 + 32) = v17;
  *(_QWORD *)(v15 + 40) = v18;
  v19 = *(_QWORD *)(a1 + 40);
  v20 = *(_BYTE *)(v19 + 32);
  v21 = *(_QWORD *)(v19 + 40);
  *(_BYTE *)(v15 + 48) = v20;
  *(_QWORD *)(v15 + 56) = v21;
  v22 = v52;
  *v52 = v10.m128i_i64[0];
  *((_DWORD *)v22 + 2) = v10.m128i_i32[2];
  v23 = v52;
  v52[2] = v11.m128i_i64[0];
  v24 = (unsigned int)v53;
  v25 = v48;
  v26 = (unsigned int)v49;
  *((_DWORD *)v23 + 6) = v11.m128i_i32[2];
  v23[4] = v12.m128i_i64[0];
  *((_DWORD *)v23 + 10) = v12.m128i_i32[2];
  *((_QWORD *)&v39 + 1) = v24;
  *(_QWORD *)&v39 = v23;
  v27 = sub_1D373B0(
          a2,
          0x2Fu,
          (__int64)&v42,
          v25,
          v26,
          *(double *)v12.m128i_i64,
          *(double *)v11.m128i_i64,
          v10,
          a6,
          v39);
  *((_QWORD *)&v38 + 1) = 2;
  v44 = v27;
  v46 = v27;
  *(_QWORD *)&v38 = &v44;
  v45 = 0;
  v47 = 1;
  v29 = sub_1D359D0(a2, 50, (__int64)&v42, 7, 0, 0, *(double *)v12.m128i_i64, *(double *)v11.m128i_i64, v10, v38);
  v30 = *(unsigned int *)(a3 + 8);
  v31 = v28;
  if ( (unsigned int)v30 >= *(_DWORD *)(a3 + 12) )
  {
    v41 = v28;
    v40 = v29;
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, (int)v29, (int)v28);
    v30 = *(unsigned int *)(a3 + 8);
    v29 = v40;
    v31 = v41;
  }
  v32 = (__int64 **)(*(_QWORD *)a3 + 16 * v30);
  *v32 = v29;
  v32[1] = v31;
  v33 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v33;
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v33 )
  {
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, (int)v29, (int)v31);
    v33 = *(unsigned int *)(a3 + 8);
  }
  v34 = (__int64 **)(*(_QWORD *)a3 + 16 * v33);
  *v34 = v27;
  v34[1] = (__int64 *)2;
  v35 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v35;
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v35 )
  {
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, (int)v29, (int)v31);
    v35 = *(unsigned int *)(a3 + 8);
  }
  v36 = (__int64 **)(*(_QWORD *)a3 + 16 * v35);
  *v36 = v27;
  v37 = v52;
  v36[1] = (__int64 *)3;
  ++*(_DWORD *)(a3 + 8);
  if ( v37 != &v54 )
    _libc_free((unsigned __int64)v37);
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
  if ( v42 )
    sub_161E7C0((__int64)&v42, v42);
}

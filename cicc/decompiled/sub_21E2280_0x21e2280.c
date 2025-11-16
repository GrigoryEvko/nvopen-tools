// Function: sub_21E2280
// Address: 0x21e2280
//
__int64 __fastcall sub_21E2280(__int64 a1, unsigned int a2, __int16 a3, __int64 a4, __m128i a5, double a6, __m128i a7)
{
  __int64 v10; // rsi
  int v11; // eax
  __int64 v12; // rcx
  int v13; // edx
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rdi
  unsigned int v17; // edx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdi
  unsigned int v24; // edx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rdi
  unsigned int v31; // edx
  __int64 v32; // rbx
  __int64 v33; // r9
  __int64 v34; // r8
  __int64 v35; // rax
  __int64 *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rbx
  const __m128i *v39; // rbx
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 v42; // rax
  const __m128i *v43; // rbx
  __int64 v44; // rax
  const __m128i *v45; // rbx
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // r12
  __int64 v50; // [rsp+8h] [rbp-288h]
  __int64 v51; // [rsp+10h] [rbp-280h]
  __int64 v52; // [rsp+10h] [rbp-280h]
  __int64 v53; // [rsp+18h] [rbp-278h]
  __int64 v54; // [rsp+18h] [rbp-278h]
  _QWORD *v55; // [rsp+20h] [rbp-270h]
  __int64 v57; // [rsp+28h] [rbp-268h]
  __int64 v58; // [rsp+30h] [rbp-260h] BYREF
  int v59; // [rsp+38h] [rbp-258h]
  __int64 v60; // [rsp+40h] [rbp-250h] BYREF
  int v61; // [rsp+48h] [rbp-248h]
  __int64 *v62; // [rsp+50h] [rbp-240h] BYREF
  __int64 v63; // [rsp+58h] [rbp-238h]
  _BYTE v64[560]; // [rsp+60h] [rbp-230h] BYREF

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL) <= 0x48u )
    sub_16BD130("bmmamma is not supported on this architecture", 1u);
  v10 = *(_QWORD *)(a4 + 72);
  v55 = *(_QWORD **)(a1 - 176);
  v58 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v58, v10, 2);
  v11 = *(_DWORD *)(a4 + 64);
  v12 = *(_QWORD *)(*(_QWORD *)(a4 + 32) + 80LL);
  v59 = v11;
  v13 = *(unsigned __int16 *)(v12 + 24);
  if ( v13 != 32 && v13 != 10 )
    sub_16BD130("rowcol not constant", 1u);
  v14 = *(_QWORD *)(v12 + 88);
  if ( *(_DWORD *)(v14 + 32) <= 0x40u )
    v53 = *(_QWORD *)(v14 + 24);
  else
    v53 = **(_QWORD **)(v14 + 24);
  v15 = *(_QWORD *)(a4 + 72);
  v62 = (__int64 *)v64;
  v63 = 0x2000000000LL;
  v60 = v15;
  if ( v15 )
  {
    sub_1623A60((__int64)&v60, v15, 2);
    v11 = *(_DWORD *)(a4 + 64);
  }
  v16 = *(_QWORD *)(a1 - 176);
  v61 = v11;
  v18 = sub_1D38BB0(v16, 2, (__int64)&v60, 5, 0, 1, a5, a6, a7, 0);
  v19 = v17;
  v20 = (unsigned int)v63;
  if ( (unsigned int)v63 >= HIDWORD(v63) )
  {
    v50 = v18;
    v52 = v17;
    sub_16CD150((__int64)&v62, v64, 0, 16, v18, v17);
    v20 = (unsigned int)v63;
    v18 = v50;
    v19 = v52;
  }
  v21 = &v62[2 * v20];
  *v21 = v18;
  v21[1] = v19;
  LODWORD(v63) = v63 + 1;
  if ( v60 )
    sub_161E7C0((__int64)&v60, v60);
  v22 = *(_QWORD *)(a4 + 72);
  v60 = v22;
  if ( v22 )
    sub_1623A60((__int64)&v60, v22, 2);
  v23 = *(_QWORD *)(a1 - 176);
  v61 = *(_DWORD *)(a4 + 64);
  v25 = sub_1D38BB0(v23, (unsigned int)v53, (__int64)&v60, 5, 0, 1, a5, a6, a7, 0);
  v26 = v24;
  v27 = (unsigned int)v63;
  if ( (unsigned int)v63 >= HIDWORD(v63) )
  {
    v51 = v25;
    v54 = v24;
    sub_16CD150((__int64)&v62, v64, 0, 16, v25, v24);
    v27 = (unsigned int)v63;
    v25 = v51;
    v26 = v54;
  }
  v28 = &v62[2 * v27];
  *v28 = v25;
  v28[1] = v26;
  LODWORD(v63) = v63 + 1;
  if ( v60 )
    sub_161E7C0((__int64)&v60, v60);
  v29 = *(_QWORD *)(a4 + 72);
  v60 = v29;
  if ( v29 )
    sub_1623A60((__int64)&v60, v29, 2);
  v30 = *(_QWORD *)(a1 - 176);
  v61 = *(_DWORD *)(a4 + 64);
  v32 = sub_1D38BB0(v30, a2, (__int64)&v60, 5, 0, 1, a5, a6, a7, 0);
  v34 = v31;
  v35 = (unsigned int)v63;
  if ( (unsigned int)v63 >= HIDWORD(v63) )
  {
    v57 = v31;
    sub_16CD150((__int64)&v62, v64, 0, 16, v31, v33);
    v35 = (unsigned int)v63;
    v34 = v57;
  }
  v36 = &v62[2 * v35];
  *v36 = v32;
  v36[1] = v34;
  v37 = (unsigned int)(v63 + 1);
  LODWORD(v63) = v63 + 1;
  if ( v60 )
  {
    sub_161E7C0((__int64)&v60, v60);
    v37 = (unsigned int)v63;
  }
  v38 = *(_QWORD *)(a4 + 32);
  if ( HIDWORD(v63) <= (unsigned int)v37 )
  {
    sub_16CD150((__int64)&v62, v64, 0, 16, v34, v33);
    v37 = (unsigned int)v63;
  }
  *(__m128i *)&v62[2 * v37] = _mm_loadu_si128((const __m128i *)(v38 + 120));
  v39 = *(const __m128i **)(a4 + 32);
  v40 = (unsigned int)(v63 + 1);
  LODWORD(v63) = v40;
  if ( HIDWORD(v63) <= (unsigned int)v40 )
  {
    sub_16CD150((__int64)&v62, v64, 0, 16, v34, v33);
    v40 = (unsigned int)v63;
  }
  *(__m128i *)&v62[2 * v40] = _mm_loadu_si128(v39 + 10);
  v41 = *(_QWORD *)(a4 + 32);
  v42 = (unsigned int)(v63 + 1);
  LODWORD(v63) = v42;
  if ( HIDWORD(v63) <= (unsigned int)v42 )
  {
    sub_16CD150((__int64)&v62, v64, 0, 16, v34, v33);
    v42 = (unsigned int)v63;
  }
  *(__m128i *)&v62[2 * v42] = _mm_loadu_si128((const __m128i *)(v41 + 200));
  v43 = *(const __m128i **)(a4 + 32);
  v44 = (unsigned int)(v63 + 1);
  LODWORD(v63) = v44;
  if ( HIDWORD(v63) <= (unsigned int)v44 )
  {
    sub_16CD150((__int64)&v62, v64, 0, 16, v34, v33);
    v44 = (unsigned int)v63;
  }
  *(__m128i *)&v62[2 * v44] = _mm_loadu_si128(v43 + 15);
  v45 = *(const __m128i **)(a4 + 32);
  v46 = (unsigned int)(v63 + 1);
  LODWORD(v63) = v46;
  if ( HIDWORD(v63) <= (unsigned int)v46 )
  {
    sub_16CD150((__int64)&v62, v64, 0, 16, v34, v33);
    v46 = (unsigned int)v63;
  }
  *(__m128i *)&v62[2 * v46] = _mm_loadu_si128(v45);
  v47 = *(_QWORD *)(a4 + 40);
  LODWORD(v63) = v63 + 1;
  v48 = sub_1D23DE0(v55, a3, (__int64)&v58, v47, *(_DWORD *)(a4 + 60), v33, v62, (unsigned int)v63);
  if ( v62 != (__int64 *)v64 )
    _libc_free((unsigned __int64)v62);
  if ( v58 )
    sub_161E7C0((__int64)&v58, v58);
  return v48;
}

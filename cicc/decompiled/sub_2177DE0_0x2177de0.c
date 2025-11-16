// Function: sub_2177DE0
// Address: 0x2177de0
//
__int64 *__fastcall sub_2177DE0(__int64 a1, __m128i a2, double a3, __m128i a4, __int64 a5, __int64 *a6)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  _QWORD *v9; // rsi
  __int64 v10; // rax
  unsigned int v11; // edx
  unsigned int v12; // eax
  __int64 v13; // r15
  _QWORD *v14; // rdx
  __int64 v15; // r9
  const __m128i *v16; // r8
  __int64 v17; // rax
  const __m128i *v18; // r15
  _QWORD *v19; // rdx
  __int64 v20; // rax
  __m128i v21; // xmm1
  __int64 v22; // rcx
  __int64 v23; // rdx
  unsigned int v24; // eax
  __int64 v25; // rdx
  bool v26; // cc
  _QWORD *v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r9
  int v32; // edx
  int v33; // esi
  unsigned __int64 v34; // rdx
  _BYTE *v35; // rcx
  __int64 v36; // rbx
  __int64 *v37; // rax
  __int64 v38; // rdx
  int v39; // r8d
  __int64 *result; // rax
  __int64 v41; // [rsp+0h] [rbp-2E0h]
  const __m128i *v42; // [rsp+8h] [rbp-2D8h]
  __int64 *v43; // [rsp+10h] [rbp-2D0h]
  __int64 *v44; // [rsp+10h] [rbp-2D0h]
  __int64 v45; // [rsp+30h] [rbp-2B0h] BYREF
  int v46; // [rsp+38h] [rbp-2A8h]
  __int64 v47; // [rsp+40h] [rbp-2A0h] BYREF
  int v48; // [rsp+48h] [rbp-298h]
  __int64 v49; // [rsp+50h] [rbp-290h]
  int v50; // [rsp+58h] [rbp-288h]
  __int64 v51; // [rsp+60h] [rbp-280h]
  int v52; // [rsp+68h] [rbp-278h]
  __int64 v53; // [rsp+70h] [rbp-270h]
  int v54; // [rsp+78h] [rbp-268h]
  __int64 *v55; // [rsp+80h] [rbp-260h]
  __int64 v56; // [rsp+88h] [rbp-258h]
  __int64 v57; // [rsp+90h] [rbp-250h]
  int v58; // [rsp+98h] [rbp-248h]
  _QWORD *v59; // [rsp+A0h] [rbp-240h] BYREF
  __int64 v60; // [rsp+A8h] [rbp-238h]
  _QWORD v61[70]; // [rsp+B0h] [rbp-230h] BYREF

  v7 = *(_QWORD *)(a1 + 72);
  v45 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v45, v7, 2);
  v46 = *(_DWORD *)(a1 + 64);
  v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 80LL) + 88LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v10 = sub_1D38BB0((__int64)a6, (__int64)v9, (__int64)&v45, 5, 0, 1, a2, a3, a4, 0);
  v59 = v61;
  v61[0] = v10;
  v61[1] = v11;
  v60 = 0x2000000001LL;
  v12 = *(_DWORD *)(a1 + 56);
  if ( v12 <= 3 )
  {
    v18 = *(const __m128i **)(a1 + 32);
    v20 = 2;
    v19 = v61;
  }
  else
  {
    v13 = 160;
    v14 = v61;
    v15 = 40LL * (v12 - 4) + 160;
    v16 = (const __m128i *)(*(_QWORD *)(a1 + 32) + 120LL);
    v17 = 1;
    while ( 1 )
    {
      a2 = _mm_loadu_si128(v16);
      *(__m128i *)&v14[2 * v17] = a2;
      v17 = (unsigned int)(v60 + 1);
      LODWORD(v60) = v60 + 1;
      if ( v15 == v13 )
        break;
      v16 = (const __m128i *)(v13 + *(_QWORD *)(a1 + 32));
      if ( HIDWORD(v60) <= (unsigned int)v17 )
      {
        v41 = v15;
        v42 = (const __m128i *)(v13 + *(_QWORD *)(a1 + 32));
        sub_16CD150((__int64)&v59, v61, 0, 16, (int)v16, v15);
        v17 = (unsigned int)v60;
        v15 = v41;
        v16 = v42;
      }
      v14 = v59;
      v13 += 40;
    }
    v18 = *(const __m128i **)(a1 + 32);
    if ( (unsigned int)v17 >= HIDWORD(v60) )
    {
      sub_16CD150((__int64)&v59, v61, 0, 16, (int)v16, v15);
      v19 = v59;
      v20 = 2LL * (unsigned int)v60;
    }
    else
    {
      v19 = v59;
      v20 = 2 * v17;
    }
  }
  v21 = _mm_loadu_si128(v18);
  *(__m128i *)&v19[v20] = v21;
  v22 = *(_QWORD *)(a1 + 32);
  v23 = *(_QWORD *)(v22 + 40);
  v24 = v60 + 1;
  LODWORD(v60) = v60 + 1;
  v25 = *(_QWORD *)(v23 + 88);
  v26 = *(_DWORD *)(v25 + 32) <= 0x40u;
  v27 = *(_QWORD **)(v25 + 24);
  if ( !v26 )
    v27 = (_QWORD *)*v27;
  if ( (_DWORD)v27 == 4424 )
  {
    v35 = *(_BYTE **)(a1 + 40);
    v34 = (unsigned __int64)v59;
    v31 = 3436;
    if ( *v35 == 5 )
      v31 = 3439;
  }
  else if ( (_DWORD)v27 == 4434 )
  {
    v35 = *(_BYTE **)(a1 + 40);
    v34 = (unsigned __int64)v59;
    v31 = 3438;
    if ( *v35 != 5 )
      v31 = 3437;
  }
  else
  {
    v28 = *(_QWORD *)(*(_QWORD *)(v22 + 360) + 88LL);
    v29 = *(_QWORD **)(v28 + 24);
    if ( *(_DWORD *)(v28 + 32) > 0x40u )
      v29 = (_QWORD *)*v29;
    v30 = sub_1D38BB0((__int64)a6, (__int64)v29, (__int64)&v45, 5, 0, 1, a2, *(double *)v21.m128i_i64, a4, 0);
    v31 = 3441;
    v33 = v32;
    v34 = (unsigned __int64)v59;
    v59[14] = v30;
    *(_DWORD *)(v34 + 120) = v33;
    v35 = *(_BYTE **)(a1 + 40);
    v24 = v60;
    if ( *v35 != 5 )
      v31 = 3440;
  }
  v36 = sub_1D23DE0(a6, v31, (__int64)&v45, (__int64)v35, *(_DWORD *)(a1 + 60), v31, (__int64 *)v34, v24);
  v37 = sub_1D3C080(a6, (__int64)&v45, v36, 4u, 2, 0, a2, *(double *)v21.m128i_i64, a4);
  v56 = v38;
  v47 = v36;
  v48 = 0;
  v49 = v36;
  v50 = 1;
  v51 = v36;
  v52 = 2;
  v53 = v36;
  v54 = 3;
  v55 = v37;
  v57 = v36;
  v58 = 5;
  result = sub_1D37190(
             (__int64)a6,
             (__int64)&v47,
             6u,
             (__int64)&v45,
             v39,
             *(double *)a2.m128i_i64,
             *(double *)v21.m128i_i64,
             a4);
  if ( v59 != v61 )
  {
    v43 = result;
    _libc_free((unsigned __int64)v59);
    result = v43;
  }
  if ( v45 )
  {
    v44 = result;
    sub_161E7C0((__int64)&v45, v45);
    return v44;
  }
  return result;
}

// Function: sub_21E2B60
// Address: 0x21e2b60
//
__int64 __fastcall sub_21E2B60(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // r12
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rax
  int v15; // edx
  _QWORD *v16; // rdx
  __int64 v17; // rcx
  _QWORD *v18; // rax
  __int64 v19; // r13
  unsigned __int64 v20; // rcx
  int v21; // edx
  __int64 v22; // rsi
  int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // r9
  int v28; // edx
  __int64 v29; // rdx
  __int64 v30; // rax
  __int16 v31; // si
  __int64 v32; // r12
  __int64 v34; // rsi
  __int64 v35; // rdx
  _QWORD *v36; // rax
  __int64 v37; // rsi
  unsigned int v38; // edx
  __int64 v39; // rdx
  __int64 v40; // rdi
  int v41; // edx
  __int64 v42; // rax
  __m128i v43; // xmm4
  __int64 v44; // rax
  __int64 v45; // rdi
  __m128i v46; // xmm7
  __int64 v47; // [rsp-8h] [rbp-108h]
  __int64 v48; // [rsp+0h] [rbp-100h]
  int v49; // [rsp+Ch] [rbp-F4h]
  _QWORD *v50; // [rsp+10h] [rbp-F0h]
  __int64 v51; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v52; // [rsp+28h] [rbp-D8h]
  __int64 v53; // [rsp+40h] [rbp-C0h] BYREF
  int v54; // [rsp+48h] [rbp-B8h]
  __int64 v55; // [rsp+50h] [rbp-B0h] BYREF
  int v56; // [rsp+58h] [rbp-A8h]
  __m128i v57; // [rsp+60h] [rbp-A0h] BYREF
  __m128i v58; // [rsp+70h] [rbp-90h]
  __int64 v59; // [rsp+80h] [rbp-80h]
  int v60; // [rsp+88h] [rbp-78h]
  __int64 v61; // [rsp+90h] [rbp-70h]
  unsigned __int64 v62; // [rsp+98h] [rbp-68h]
  __int64 v63; // [rsp+A0h] [rbp-60h]
  int v64; // [rsp+A8h] [rbp-58h]
  __m128i v65; // [rsp+B0h] [rbp-50h]
  __m128i v66; // [rsp+C0h] [rbp-40h]

  v50 = *(_QWORD **)(a1 - 176);
  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 88LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v10 = *(_QWORD *)(*(_QWORD *)(v7 + 80) + 88LL);
  v11 = *(_QWORD **)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = (_QWORD *)*v11;
  v12 = *(_QWORD *)(a2 + 72);
  v57.m128i_i64[0] = *(_QWORD *)(a2 + 72);
  if ( ((unsigned __int8)v11 & 1) != 0 )
  {
    if ( v12 )
    {
      sub_1623A60((__int64)&v57, v12, 2);
      v13 = *(_QWORD *)(a1 - 176);
    }
    else
    {
      v13 = (__int64)v50;
    }
    v57.m128i_i32[2] = *(_DWORD *)(a2 + 64);
    v14 = sub_1D38BB0(v13, 1, (__int64)&v57, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
  }
  else
  {
    if ( v12 )
    {
      sub_1623A60((__int64)&v57, v12, 2);
      v45 = *(_QWORD *)(a1 - 176);
    }
    else
    {
      v45 = (__int64)v50;
    }
    v57.m128i_i32[2] = *(_DWORD *)(a2 + 64);
    v14 = sub_1D38BB0(v45, 0, (__int64)&v57, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
  }
  v48 = v14;
  v49 = v15;
  if ( v57.m128i_i64[0] )
    sub_161E7C0((__int64)&v57, v57.m128i_i64[0]);
  v16 = *(_QWORD **)(a2 + 32);
  v17 = *(_QWORD *)(v16[25] + 88LL);
  v18 = *(_QWORD **)(v17 + 24);
  if ( *(_DWORD *)(v17 + 32) > 0x40u )
    v18 = (_QWORD *)*v18;
  v19 = v16[30];
  v20 = v16[31];
  v51 = (unsigned int)v18;
  v21 = *(unsigned __int16 *)(v19 + 24);
  v52 = v20;
  if ( v21 != 32 && v21 != 10 )
    goto LABEL_15;
  v34 = *(_QWORD *)(a2 + 72);
  v57.m128i_i64[0] = v34;
  if ( v34 )
    sub_1623A60((__int64)&v57, v34, 2);
  v57.m128i_i32[2] = *(_DWORD *)(a2 + 64);
  v35 = *(_QWORD *)(v19 + 88);
  v36 = *(_QWORD **)(v35 + 24);
  if ( *(_DWORD *)(v35 + 32) > 0x40u )
    v36 = (_QWORD *)*v36;
  v37 = (unsigned int)v36;
  if ( v36 == (_QWORD *)v51 )
    v37 = 0xFFFFFFFFLL;
  v19 = sub_1D38BB0(*(_QWORD *)(a1 - 176), v37, (__int64)&v57, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
  v52 = v38 | v52 & 0xFFFFFFFF00000000LL;
  if ( !v57.m128i_i64[0] )
  {
LABEL_15:
    v22 = *(_QWORD *)(a2 + 72);
    v53 = v22;
    if ( !v22 )
      goto LABEL_34;
LABEL_16:
    sub_1623A60((__int64)&v53, v22, 2);
    v23 = *(_DWORD *)(a2 + 64);
    v24 = *(_QWORD *)(a2 + 72);
    v25 = *(_QWORD *)(a2 + 32);
    v54 = v23;
    if ( (_DWORD)v9 == 4166 )
    {
      v57 = _mm_loadu_si128((const __m128i *)(v25 + 120));
      v46 = _mm_loadu_si128((const __m128i *)(v25 + 160));
      v55 = v24;
      v58 = v46;
      if ( v24 )
      {
        sub_1623A60((__int64)&v55, v24, 2);
        v23 = *(_DWORD *)(a2 + 64);
      }
      goto LABEL_35;
    }
    a4 = _mm_loadu_si128((const __m128i *)(v25 + 120));
    v57 = a4;
    a5 = _mm_loadu_si128((const __m128i *)(v25 + 160));
    v55 = v24;
    v58 = a5;
    if ( v24 )
    {
      sub_1623A60((__int64)&v55, v24, 2);
      v23 = *(_DWORD *)(a2 + 64);
    }
LABEL_19:
    v26 = *(_QWORD *)(a1 - 176);
    v56 = v23;
    v59 = sub_1D38BB0(v26, v51, (__int64)&v55, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v60 = v28;
    if ( v55 )
      sub_161E7C0((__int64)&v55, v55);
    v61 = v19;
    v29 = 6;
    v62 = v52;
    v63 = v48;
    v64 = v49;
    v30 = *(_QWORD *)(a1 + 16);
    v65 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
    v31 = 3141 - ((*(_BYTE *)(v30 + 936) == 0) - 1);
    goto LABEL_22;
  }
  sub_161E7C0((__int64)&v57, v57.m128i_i64[0]);
  v22 = *(_QWORD *)(a2 + 72);
  v53 = v22;
  if ( v22 )
    goto LABEL_16;
LABEL_34:
  v39 = *(_QWORD *)(a2 + 32);
  v23 = *(_DWORD *)(a2 + 64);
  v54 = v23;
  v57 = _mm_loadu_si128((const __m128i *)(v39 + 120));
  v55 = 0;
  v58 = _mm_loadu_si128((const __m128i *)(v39 + 160));
  if ( (_DWORD)v9 != 4166 )
    goto LABEL_19;
LABEL_35:
  v40 = *(_QWORD *)(a1 - 176);
  v56 = v23;
  v59 = sub_1D38BB0(v40, v51, (__int64)&v55, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
  v27 = v47;
  v60 = v41;
  if ( v55 )
    sub_161E7C0((__int64)&v55, v55);
  v61 = v19;
  v29 = 7;
  v62 = v52;
  v63 = v48;
  v64 = v49;
  v42 = *(_QWORD *)(a2 + 32);
  v65 = _mm_loadu_si128((const __m128i *)(v42 + 280));
  v43 = _mm_loadu_si128((const __m128i *)v42);
  v44 = *(_QWORD *)(a1 + 16);
  v66 = v43;
  v31 = 3145 - ((*(_BYTE *)(v44 + 936) == 0) - 1);
LABEL_22:
  v32 = sub_1D23DE0(v50, v31, (__int64)&v53, *(_QWORD *)(a2 + 40), *(_DWORD *)(a2 + 60), v27, v57.m128i_i64, v29);
  if ( v53 )
    sub_161E7C0((__int64)&v53, v53);
  return v32;
}

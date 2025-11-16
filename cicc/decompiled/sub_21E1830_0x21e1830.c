// Function: sub_21E1830
// Address: 0x21e1830
//
__int64 __fastcall sub_21E1830(__int64 a1, int a2, __int64 a3, __m128i a4, double a5, __m128i a6)
{
  unsigned int v6; // eax
  __int16 v9; // r12
  __int64 v10; // rsi
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rax
  unsigned int v14; // eax
  int v15; // r8d
  const __m128i *v16; // r9
  __m128i v17; // xmm1
  __int64 v18; // rsi
  unsigned int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // edx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rax
  __int64 v27; // r8
  __m128i v28; // xmm0
  __int64 v29; // rsi
  __int64 v30; // rdi
  unsigned int v31; // edx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rax
  __int64 *v35; // rax
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 v38; // r8
  const __m128i *v39; // r14
  const __m128i *v40; // r15
  __int64 v41; // rcx
  __int64 v42; // r12
  __int64 v44; // [rsp+8h] [rbp-288h]
  unsigned int v45; // [rsp+10h] [rbp-280h]
  __int64 v46; // [rsp+10h] [rbp-280h]
  __int64 v47; // [rsp+10h] [rbp-280h]
  __int64 v48; // [rsp+10h] [rbp-280h]
  __int64 v49; // [rsp+18h] [rbp-278h]
  __int64 v50; // [rsp+18h] [rbp-278h]
  __int64 v51; // [rsp+18h] [rbp-278h]
  _QWORD *v52; // [rsp+20h] [rbp-270h]
  int v53; // [rsp+28h] [rbp-268h]
  __int64 v54; // [rsp+30h] [rbp-260h] BYREF
  int v55; // [rsp+38h] [rbp-258h]
  __int64 v56; // [rsp+40h] [rbp-250h] BYREF
  int v57; // [rsp+48h] [rbp-248h]
  __int64 *v58; // [rsp+50h] [rbp-240h] BYREF
  __int64 v59; // [rsp+58h] [rbp-238h]
  _OWORD v60[35]; // [rsp+60h] [rbp-230h] BYREF

  v6 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL);
  if ( v6 <= 0x47 )
    goto LABEL_40;
  v9 = a2;
  if ( (unsigned int)(a2 - 610) <= 1 || (unsigned int)(a2 - 179) <= 1 )
  {
    if ( v6 != 72 )
    {
      v53 = 2;
      goto LABEL_6;
    }
LABEL_40:
    sub_16BD130("imma stc not supported on this architecture", 1u);
  }
  v53 = 8;
LABEL_6:
  v10 = *(_QWORD *)(a3 + 72);
  v52 = *(_QWORD **)(a1 - 176);
  v54 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v54, v10, 2);
  v11 = *(_QWORD *)(*(_QWORD *)(a3 + 32) + 160LL);
  v55 = *(_DWORD *)(a3 + 64);
  v12 = *(unsigned __int16 *)(v11 + 24);
  if ( v12 != 10 && v12 != 32 )
    sub_16BD130("rowcol not constant", 1u);
  v13 = *(_QWORD *)(v11 + 88);
  if ( *(_DWORD *)(v13 + 32) <= 0x40u )
    v49 = *(_QWORD *)(v13 + 24);
  else
    v49 = **(_QWORD **)(v13 + 24);
  v14 = sub_21DEF90(a3);
  v17 = _mm_loadu_si128(v16 + 5);
  v18 = *(_QWORD *)(a3 + 72);
  v58 = (__int64 *)v60;
  v19 = v14;
  v59 = 0x2000000001LL;
  v56 = v18;
  v60[0] = v17;
  if ( v18 )
  {
    v45 = v14;
    sub_1623A60((__int64)&v56, v18, 2);
    v15 = *(_DWORD *)(a3 + 64);
    v19 = v45;
  }
  v20 = *(_QWORD *)(a1 - 176);
  v57 = v15;
  v22 = sub_1D38BB0(v20, v19, (__int64)&v56, 5, 0, 1, a4, *(double *)v17.m128i_i64, a6, 0);
  v23 = v21;
  v24 = (unsigned int)v59;
  if ( (unsigned int)v59 >= HIDWORD(v59) )
  {
    v44 = v22;
    v48 = v21;
    sub_16CD150((__int64)&v58, v60, 0, 16, v22, v21);
    v24 = (unsigned int)v59;
    v22 = v44;
    v23 = v48;
  }
  v25 = &v58[2 * v24];
  *v25 = v22;
  v25[1] = v23;
  v26 = (unsigned int)(v59 + 1);
  LODWORD(v59) = v59 + 1;
  if ( v56 )
  {
    sub_161E7C0((__int64)&v56, v56);
    v26 = (unsigned int)v59;
  }
  v27 = *(_QWORD *)(a3 + 32);
  if ( (unsigned int)v26 >= HIDWORD(v59) )
  {
    v47 = *(_QWORD *)(a3 + 32);
    sub_16CD150((__int64)&v58, v60, 0, 16, v27, v23);
    v26 = (unsigned int)v59;
    v27 = v47;
  }
  v28 = _mm_loadu_si128((const __m128i *)(v27 + 120));
  *(__m128i *)&v58[2 * v26] = v28;
  v29 = *(_QWORD *)(a3 + 72);
  LODWORD(v59) = v59 + 1;
  v56 = v29;
  if ( v29 )
    sub_1623A60((__int64)&v56, v29, 2);
  v30 = *(_QWORD *)(a1 - 176);
  v57 = *(_DWORD *)(a3 + 64);
  v32 = sub_1D38BB0(v30, (unsigned int)v49, (__int64)&v56, 5, 0, 1, v28, *(double *)v17.m128i_i64, a6, 0);
  v33 = v31;
  v34 = (unsigned int)v59;
  if ( (unsigned int)v59 >= HIDWORD(v59) )
  {
    v46 = v32;
    v51 = v31;
    sub_16CD150((__int64)&v58, v60, 0, 16, v32, v31);
    v34 = (unsigned int)v59;
    v32 = v46;
    v33 = v51;
  }
  v35 = &v58[2 * v34];
  *v35 = v32;
  v35[1] = v33;
  v36 = (unsigned int)(v59 + 1);
  LODWORD(v59) = v59 + 1;
  if ( v56 )
  {
    sub_161E7C0((__int64)&v56, v56);
    v36 = (unsigned int)v59;
  }
  v37 = 200;
  v38 = 40LL * (unsigned int)(v53 - 1) + 240;
  do
  {
    v39 = (const __m128i *)(v37 + *(_QWORD *)(a3 + 32));
    if ( HIDWORD(v59) <= (unsigned int)v36 )
    {
      v50 = v38;
      sub_16CD150((__int64)&v58, v60, 0, 16, v38, v33);
      v36 = (unsigned int)v59;
      v38 = v50;
    }
    v37 += 40;
    *(__m128i *)&v58[2 * v36] = _mm_loadu_si128(v39);
    v36 = (unsigned int)(v59 + 1);
    LODWORD(v59) = v59 + 1;
  }
  while ( v38 != v37 );
  v40 = *(const __m128i **)(a3 + 32);
  if ( (unsigned int)v36 >= HIDWORD(v59) )
  {
    sub_16CD150((__int64)&v58, v60, 0, 16, v38, v33);
    v36 = (unsigned int)v59;
  }
  *(__m128i *)&v58[2 * v36] = _mm_loadu_si128(v40);
  v41 = *(_QWORD *)(a3 + 40);
  LODWORD(v59) = v59 + 1;
  v42 = sub_1D23DE0(v52, v9, (__int64)&v54, v41, *(_DWORD *)(a3 + 60), v33, v58, (unsigned int)v59);
  if ( v58 != (__int64 *)v60 )
    _libc_free((unsigned __int64)v58);
  if ( v54 )
    sub_161E7C0((__int64)&v54, v54);
  return v42;
}

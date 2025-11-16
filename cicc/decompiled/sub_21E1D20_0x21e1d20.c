// Function: sub_21E1D20
// Address: 0x21e1d20
//
__int64 __fastcall sub_21E1D20(__int64 a1, unsigned int a2, int a3, __int64 a4, __m128i a5, double a6, __m128i a7)
{
  unsigned int v7; // eax
  __int64 v12; // rsi
  __int64 v13; // rdx
  int v14; // edi
  __int64 v15; // rcx
  int v16; // eax
  __int64 v17; // rax
  _QWORD *v18; // rcx
  __int64 v19; // rdx
  int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rsi
  unsigned int v23; // edx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rdi
  unsigned int v30; // edx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rdi
  unsigned int v37; // edx
  __int64 v38; // rbx
  __int64 v39; // r9
  __int64 v40; // r8
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rax
  int v44; // edx
  __int64 v45; // rbx
  __int64 v46; // r13
  const __m128i *v47; // r14
  const __m128i *v48; // rbx
  __int64 v49; // r8
  __int64 v50; // r12
  __int64 v52; // [rsp+8h] [rbp-288h]
  int v53; // [rsp+10h] [rbp-280h]
  __int64 v54; // [rsp+10h] [rbp-280h]
  __int64 v55; // [rsp+10h] [rbp-280h]
  __int64 v56; // [rsp+18h] [rbp-278h]
  __int64 v57; // [rsp+18h] [rbp-278h]
  __int64 v58; // [rsp+18h] [rbp-278h]
  _QWORD *v59; // [rsp+28h] [rbp-268h]
  __int64 v60; // [rsp+30h] [rbp-260h] BYREF
  int v61; // [rsp+38h] [rbp-258h]
  __int64 v62; // [rsp+40h] [rbp-250h] BYREF
  int v63; // [rsp+48h] [rbp-248h]
  __int64 *v64; // [rsp+50h] [rbp-240h] BYREF
  __int64 v65; // [rsp+58h] [rbp-238h]
  _BYTE v66[560]; // [rsp+60h] [rbp-230h] BYREF

  v7 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL);
  if ( v7 <= 0x47 || a2 > 1 && v7 == 72 )
    sub_16BD130("immamma is not supported on this architecture", 1u);
  v12 = *(_QWORD *)(a4 + 72);
  v59 = *(_QWORD **)(a1 - 176);
  v60 = v12;
  if ( v12 )
    sub_1623A60((__int64)&v60, v12, 2);
  v13 = *(_QWORD *)(a4 + 32);
  v14 = *(_DWORD *)(a4 + 64);
  v15 = *(_QWORD *)(v13 + 80);
  v61 = v14;
  v16 = *(unsigned __int16 *)(v15 + 24);
  if ( v16 != 10 && v16 != 32 )
    sub_16BD130("rowcol not constant", 1u);
  v17 = *(_QWORD *)(v15 + 88);
  v18 = *(_QWORD **)(v17 + 24);
  if ( *(_DWORD *)(v17 + 32) > 0x40u )
    v18 = (_QWORD *)*v18;
  v19 = *(_QWORD *)(v13 + 120);
  v20 = *(unsigned __int16 *)(v19 + 24);
  if ( v20 != 32 && v20 != 10 )
    sub_16BD130("satf not constant", 1u);
  v21 = *(_QWORD *)(v19 + 88);
  if ( *(_DWORD *)(v21 + 32) <= 0x40u )
    v56 = *(_QWORD *)(v21 + 24);
  else
    v56 = **(_QWORD **)(v21 + 24);
  v22 = *(_QWORD *)(a4 + 72);
  v64 = (__int64 *)v66;
  v65 = 0x2000000000LL;
  v62 = v22;
  if ( v22 )
  {
    v53 = (int)v18;
    sub_1623A60((__int64)&v62, v22, 2);
    v14 = *(_DWORD *)(a4 + 64);
    LODWORD(v18) = v53;
  }
  v63 = v14;
  v24 = sub_1D38BB0(*(_QWORD *)(a1 - 176), (unsigned int)v18, (__int64)&v62, 5, 0, 1, a5, a6, a7, 0);
  v25 = v23;
  v26 = (unsigned int)v65;
  if ( (unsigned int)v65 >= HIDWORD(v65) )
  {
    v52 = v24;
    v54 = v23;
    sub_16CD150((__int64)&v64, v66, 0, 16, v24, v23);
    v26 = (unsigned int)v65;
    v24 = v52;
    v25 = v54;
  }
  v27 = &v64[2 * v26];
  *v27 = v24;
  v27[1] = v25;
  LODWORD(v65) = v65 + 1;
  if ( v62 )
    sub_161E7C0((__int64)&v62, v62);
  v28 = *(_QWORD *)(a4 + 72);
  v62 = v28;
  if ( v28 )
    sub_1623A60((__int64)&v62, v28, 2);
  v29 = *(_QWORD *)(a1 - 176);
  v63 = *(_DWORD *)(a4 + 64);
  v31 = sub_1D38BB0(v29, (unsigned int)v56, (__int64)&v62, 5, 0, 1, a5, a6, a7, 0);
  v32 = v30;
  v33 = (unsigned int)v65;
  if ( (unsigned int)v65 >= HIDWORD(v65) )
  {
    v55 = v31;
    v57 = v30;
    sub_16CD150((__int64)&v64, v66, 0, 16, v31, v30);
    v33 = (unsigned int)v65;
    v31 = v55;
    v32 = v57;
  }
  v34 = &v64[2 * v33];
  *v34 = v31;
  v34[1] = v32;
  LODWORD(v65) = v65 + 1;
  if ( v62 )
    sub_161E7C0((__int64)&v62, v62);
  v35 = *(_QWORD *)(a4 + 72);
  v62 = v35;
  if ( v35 )
    sub_1623A60((__int64)&v62, v35, 2);
  v36 = *(_QWORD *)(a1 - 176);
  v63 = *(_DWORD *)(a4 + 64);
  v38 = sub_1D38BB0(v36, a2, (__int64)&v62, 5, 0, 1, a5, a6, a7, 0);
  v40 = v37;
  v41 = (unsigned int)v65;
  if ( (unsigned int)v65 >= HIDWORD(v65) )
  {
    v58 = v37;
    sub_16CD150((__int64)&v64, v66, 0, 16, v37, v39);
    v41 = (unsigned int)v65;
    v40 = v58;
  }
  v42 = &v64[2 * v41];
  *v42 = v38;
  v42[1] = v40;
  v43 = (unsigned int)(v65 + 1);
  LODWORD(v65) = v65 + 1;
  if ( v62 )
  {
    sub_161E7C0((__int64)&v62, v62);
    v43 = (unsigned int)v65;
  }
  v44 = 12;
  if ( a3 != 584 )
    v44 = 9 * (a3 != 609) + 4;
  v45 = 160;
  v46 = 40LL * (unsigned int)(v44 - 1) + 200;
  do
  {
    v47 = (const __m128i *)(v45 + *(_QWORD *)(a4 + 32));
    if ( HIDWORD(v65) <= (unsigned int)v43 )
    {
      sub_16CD150((__int64)&v64, v66, 0, 16, v40, v39);
      v43 = (unsigned int)v65;
    }
    v45 += 40;
    *(__m128i *)&v64[2 * v43] = _mm_loadu_si128(v47);
    v43 = (unsigned int)(v65 + 1);
    LODWORD(v65) = v65 + 1;
  }
  while ( v46 != v45 );
  v48 = *(const __m128i **)(a4 + 32);
  if ( (unsigned int)v43 >= HIDWORD(v65) )
  {
    sub_16CD150((__int64)&v64, v66, 0, 16, v40, v39);
    v43 = (unsigned int)v65;
  }
  *(__m128i *)&v64[2 * v43] = _mm_loadu_si128(v48);
  v49 = *(_QWORD *)(a4 + 40);
  LODWORD(v65) = v65 + 1;
  v50 = sub_1D23DE0(v59, a3, (__int64)&v60, v49, *(_DWORD *)(a4 + 60), v39, v64, (unsigned int)v65);
  if ( v64 != (__int64 *)v66 )
    _libc_free((unsigned __int64)v64);
  if ( v60 )
    sub_161E7C0((__int64)&v60, v60);
  return v50;
}

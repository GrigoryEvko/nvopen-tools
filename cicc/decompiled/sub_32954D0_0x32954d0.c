// Function: sub_32954D0
// Address: 0x32954d0
//
__int64 __fastcall sub_32954D0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r13
  int v6; // r14d
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rcx
  __int64 *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 result; // rax
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdi
  int v17; // eax
  bool v18; // dl
  unsigned __int16 *v19; // rax
  char v20; // r8
  __int64 v21; // rcx
  __int64 v22; // r8
  unsigned int v23; // eax
  __m128i v24; // xmm6
  __m128i v25; // xmm7
  __int64 v26; // r14
  int v27; // ebx
  __int64 v28; // rax
  int v29; // eax
  int v30; // edx
  __int64 v31; // r8
  __m128i v32; // xmm4
  __m128i v33; // xmm5
  int v34; // [rsp+0h] [rbp-120h]
  __int64 v35; // [rsp+8h] [rbp-118h]
  int v36; // [rsp+14h] [rbp-10Ch]
  __int64 v37; // [rsp+18h] [rbp-108h]
  int v38; // [rsp+20h] [rbp-100h]
  int v39; // [rsp+24h] [rbp-FCh]
  __int64 v40; // [rsp+28h] [rbp-F8h]
  __int16 v41; // [rsp+28h] [rbp-F8h]
  unsigned int v42; // [rsp+30h] [rbp-F0h]
  __int64 v43; // [rsp+30h] [rbp-F0h]
  __int64 v44; // [rsp+38h] [rbp-E8h]
  __int64 v45; // [rsp+38h] [rbp-E8h]
  int v46; // [rsp+38h] [rbp-E8h]
  int v47; // [rsp+4Ch] [rbp-D4h] BYREF
  __m128i v48; // [rsp+50h] [rbp-D0h] BYREF
  __m128i v49; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+70h] [rbp-B0h] BYREF
  int v51; // [rsp+78h] [rbp-A8h]
  __int64 v52; // [rsp+80h] [rbp-A0h] BYREF
  int v53; // [rsp+88h] [rbp-98h]
  __int64 v54; // [rsp+90h] [rbp-90h]
  unsigned int v55; // [rsp+98h] [rbp-88h]
  __m128i v56; // [rsp+A0h] [rbp-80h]
  __m128i v57; // [rsp+B0h] [rbp-70h]
  __int64 v58; // [rsp+C0h] [rbp-60h]
  int v59; // [rsp+C8h] [rbp-58h]
  __int64 v60; // [rsp+D0h] [rbp-50h]
  int v61; // [rsp+D8h] [rbp-48h]
  __int64 v62; // [rsp+E0h] [rbp-40h]
  int v63; // [rsp+E8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)v4;
  v6 = *(_DWORD *)(v4 + 8);
  if ( *(_DWORD *)(a2 + 24) == 470 )
  {
    v7 = 200;
    v36 = *(_DWORD *)(v4 + 168);
    v38 = *(_DWORD *)(v4 + 128);
    v37 = *(_QWORD *)(v4 + 120);
    v42 = *(_DWORD *)(v4 + 48);
    v8 = *(_QWORD *)(v4 + 160);
    v44 = *(_QWORD *)(v4 + 40);
    v48 = _mm_loadu_si128((const __m128i *)(v4 + 80));
    v49 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  }
  else
  {
    v7 = 240;
    v8 = *(_QWORD *)(v4 + 200);
    v36 = *(_DWORD *)(v4 + 208);
    v9 = *(_QWORD *)(v4 + 160);
    v48 = _mm_loadu_si128((const __m128i *)(v4 + 120));
    v37 = v9;
    LODWORD(v9) = *(_DWORD *)(v4 + 168);
    v49 = _mm_loadu_si128((const __m128i *)(v4 + 80));
    v38 = v9;
    v44 = *(_QWORD *)(v4 + 40);
    v42 = *(_DWORD *)(v4 + 48);
  }
  v10 = (__int64 *)(v7 + v4);
  v11 = *(_QWORD *)(a2 + 80);
  v12 = *v10;
  LODWORD(v10) = *((_DWORD *)v10 + 2);
  v50 = v11;
  v39 = (int)v10;
  v40 = v12;
  v47 = (*(_WORD *)(a2 + 32) >> 7) & 7;
  if ( v11 )
    sub_B96E90((__int64)&v50, v11, 1);
  v51 = *(_DWORD *)(a2 + 72);
  if ( (unsigned __int8)sub_33D1AE0(v8, 0) )
  {
    result = v5;
    goto LABEL_7;
  }
  v14 = 120;
  v15 = *a1;
  if ( *(_DWORD *)(a2 + 24) != 470 )
    v14 = 160;
  v16 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + v14) + 96LL);
  if ( *(_DWORD *)(v16 + 32) <= 0x40u )
  {
    v18 = *(_QWORD *)(v16 + 24) == 1;
  }
  else
  {
    v34 = *(_DWORD *)(v16 + 32);
    v35 = *a1;
    v17 = sub_C444A0(v16 + 24);
    v15 = v35;
    v18 = v34 - 1 == v17;
  }
  if ( (unsigned __int8)sub_32945F0((unsigned int *)&v49, (unsigned int *)&v48, !v18, v15, (int)&v50) )
  {
    v52 = v5;
    v31 = *(_QWORD *)(a2 + 104);
    v60 = v8;
    v54 = v44;
    v59 = v38;
    v55 = v42;
    v32 = _mm_loadu_si128(&v49);
    v63 = v39;
    v33 = _mm_loadu_si128(&v48);
    v21 = *(_QWORD *)(a2 + 112);
    v58 = v37;
    v53 = v6;
    v61 = v36;
    v26 = *a1;
    v62 = v40;
    v27 = *(unsigned __int16 *)(a2 + 96);
    v41 = v47;
    v43 = v21;
    v46 = v31;
    v56 = v32;
    v57 = v33;
  }
  else
  {
    v19 = (unsigned __int16 *)(*(_QWORD *)(v44 + 48) + 16LL * v42);
    v20 = sub_3294A30(&v48, &v47, *v19, *((_QWORD *)v19 + 1), *a1);
    result = 0;
    if ( !v20 )
      goto LABEL_7;
    v60 = v8;
    v21 = *(_QWORD *)(a2 + 112);
    v22 = *(_QWORD *)(a2 + 104);
    v53 = v6;
    v54 = v44;
    v23 = v42;
    v24 = _mm_loadu_si128(&v49);
    v25 = _mm_loadu_si128(&v48);
    v43 = v21;
    v55 = v23;
    v46 = v22;
    v26 = *a1;
    v58 = v37;
    v27 = *(unsigned __int16 *)(a2 + 96);
    v52 = v5;
    v59 = v38;
    v56 = v24;
    v61 = v36;
    v28 = v40;
    v41 = v47;
    v62 = v28;
    v57 = v25;
    v63 = v39;
  }
  v29 = sub_33ED250(v26, 1, 0, v21);
  result = sub_33E6FD0(v26, v29, v30, v27, v46, (unsigned int)&v50, (__int64)&v52, 7, v43, v41);
LABEL_7:
  if ( v50 )
  {
    v45 = result;
    sub_B91220((__int64)&v50, v50);
    return v45;
  }
  return result;
}

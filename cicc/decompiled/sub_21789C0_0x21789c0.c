// Function: sub_21789C0
// Address: 0x21789c0
//
__int64 *__fastcall sub_21789C0(__int64 a1, unsigned int a2, __int64 *a3, __m128i a4)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int128 v8; // xmm1
  __int128 v9; // xmm2
  unsigned __int8 *v10; // rsi
  const void **v11; // rbx
  __int64 *v12; // r12
  char v14; // al
  __int16 v15; // r10
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // r14
  __int128 v22; // rax
  __int64 v23; // r9
  __int64 v24; // rax
  unsigned int v25; // edx
  int v26; // r15d
  __int64 v27; // rcx
  __int64 v28; // r9
  _QWORD *v29; // rsi
  unsigned int v30; // edx
  unsigned __int64 v31; // rax
  int v32; // edx
  __int64 v33; // r15
  __int16 v34; // r14
  __int16 v35; // r14
  unsigned int v36; // edx
  __int16 v37; // si
  __int128 v38; // rax
  __int64 v39; // r9
  __int64 v40; // rax
  __int128 v41; // rax
  __int64 *v42; // r14
  __int64 v43; // rdx
  __int64 v44; // r15
  __int128 v45; // rax
  __int128 v46; // [rsp-20h] [rbp-F0h]
  __int128 v47; // [rsp-10h] [rbp-E0h]
  int v48; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v49; // [rsp+8h] [rbp-C8h]
  __int16 v50; // [rsp+10h] [rbp-C0h]
  unsigned __int8 v51; // [rsp+20h] [rbp-B0h]
  __int128 v52; // [rsp+30h] [rbp-A0h]
  unsigned int v53; // [rsp+40h] [rbp-90h]
  char v54; // [rsp+4Bh] [rbp-85h]
  int v55; // [rsp+4Ch] [rbp-84h]
  __int128 v56; // [rsp+50h] [rbp-80h]
  __int64 v57; // [rsp+90h] [rbp-40h] BYREF
  int v58; // [rsp+98h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 32);
  v7 = *(_QWORD *)(a1 + 72);
  v8 = (__int128)_mm_loadu_si128((const __m128i *)v6);
  v9 = (__int128)_mm_loadu_si128((const __m128i *)(v6 + 40));
  v57 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v57, v7, 2);
  v10 = (unsigned __int8 *)(*(_QWORD *)(a1 + 40) + 16LL * a2);
  v58 = *(_DWORD *)(a1 + 64);
  v11 = (const void **)*((_QWORD *)v10 + 1);
  v55 = v58;
  v51 = *v10;
  v53 = *v10;
  if ( *v10 == 9 )
  {
    v14 = sub_21CF380(a3[2], a3[4]);
    v15 = 3349;
    v54 = v14;
    v16 = -(__int64)(v14 == 0);
    LOBYTE(v16) = 0;
    v17 = v16 + 258;
  }
  else
  {
    if ( *v10 != 10 )
    {
      v12 = 0;
      goto LABEL_6;
    }
    sub_21CF380(a3[2], a3[4]);
    v17 = 2;
    v54 = 0;
    v15 = 3352;
  }
  v50 = v15;
  v18 = sub_1D38BB0((__int64)a3, v17, (__int64)&v57, 5, 0, 1, a4, *(double *)&v8, (__m128i)v9, 0);
  v20 = v19;
  v21 = v18;
  *(_QWORD *)&v22 = sub_1D364E0((__int64)a3, (__int64)&v57, v51, v11, 1u, 0.0, *(double *)&v8, (__m128i)v9);
  *((_QWORD *)&v46 + 1) = v20;
  *(_QWORD *)&v46 = v21;
  v24 = sub_1D2CD40(a3, v50, (__int64)&v57, 2, 0, v23, v8, v22, v46);
  *(_DWORD *)(v24 + 64) = v55;
  v49 = v24;
  v26 = v55 + 2;
  *(_QWORD *)&v52 = sub_1D309E0(a3, 163, (__int64)&v57, v53, v11, 0, 0.0, *(double *)&v8, *(double *)&v9, v8);
  *(_DWORD *)(v52 + 64) = v55 + 1;
  *((_QWORD *)&v52 + 1) = v25 | *((_QWORD *)&v8 + 1) & 0xFFFFFFFF00000000LL;
  v27 = sub_1D309E0(a3, 163, (__int64)&v57, v53, v11, 0, 0.0, *(double *)&v8, *(double *)&v9, v9);
  v29 = (_QWORD *)v27;
  *(_QWORD *)&v56 = v27;
  v48 = v55 + 3;
  v31 = v30 | *((_QWORD *)&v9 + 1) & 0xFFFFFFFF00000000LL;
  v32 = *(unsigned __int16 *)(v27 + 24);
  *((_QWORD *)&v56 + 1) = v31;
  if ( (_WORD)v32 == 33 || v32 == 11 )
  {
    v35 = 439 - ((v54 == 0) - 1);
    if ( v51 != 9 )
      v35 = 446;
    if ( (_WORD)v32 != 33 )
    {
      v29 = sub_1D360F0(a3, *(_QWORD *)(v27 + 88), (__int64)&v57, v51, v11, 1, 0.0, *(double *)&v8, (__m128i)v9);
      *((_QWORD *)&v56 + 1) = v36 | *((_QWORD *)&v56 + 1) & 0xFFFFFFFF00000000LL;
    }
    *((_DWORD *)v29 + 16) = v26;
    *(_QWORD *)&v56 = v29;
    v37 = v35;
    v34 = 231;
    v33 = sub_1D2CCE0(a3, v37, (__int64)&v57, v53, (__int64)v11, v28, v52, v56);
    *(_DWORD *)(v33 + 64) = v48;
    if ( v51 == 9 )
      v34 = 219;
  }
  else
  {
    *(_DWORD *)(v27 + 64) = v26;
    if ( v51 == 9 )
    {
      v33 = sub_1D2CCE0(a3, 443 - ((v54 == 0) - 1), (__int64)&v57, v53, (__int64)v11, v28, v52, v56);
      v34 = 219;
    }
    else
    {
      v33 = sub_1D2CCE0(a3, 447, (__int64)&v57, v53, (__int64)v11, v28, v52, v56);
      v34 = 231;
    }
    *(_DWORD *)(v33 + 64) = v48;
  }
  *(_QWORD *)&v38 = sub_1D38BB0(
                      (__int64)a3,
                      (-(__int64)(v54 == 0) & 0xFFFFFFFFFFFFFFF0LL) + 19,
                      (__int64)&v57,
                      5,
                      0,
                      1,
                      (__m128i)0LL,
                      *(double *)&v8,
                      (__m128i)v9,
                      0);
  v40 = sub_1D2CCE0(a3, v34, (__int64)&v57, v53, (__int64)v11, v39, (unsigned __int64)v33, v38);
  *(_DWORD *)(v40 + 64) = v55 + 4;
  *(_QWORD *)&v41 = sub_1D332F0(a3, 78, (__int64)&v57, v53, v11, 0, 0.0, *(double *)&v8, (__m128i)v9, v40, 0, v56);
  *(_DWORD *)(v41 + 64) = v55 + 5;
  v42 = sub_1D332F0(
          a3,
          77,
          (__int64)&v57,
          v53,
          v11,
          0,
          0.0,
          *(double *)&v8,
          (__m128i)v9,
          v52,
          *((unsigned __int64 *)&v52 + 1),
          v41);
  v44 = v43;
  *((_DWORD *)v42 + 16) = v55 + 6;
  *((_QWORD *)&v47 + 1) = v43;
  *(_QWORD *)&v47 = v42;
  *(_QWORD *)&v45 = sub_1D309E0(a3, 162, (__int64)&v57, v53, v11, 0, 0.0, *(double *)&v8, *(double *)&v9, v47);
  *(_DWORD *)(v45 + 64) = v55 + 7;
  v12 = sub_1D3A900(
          a3,
          0x86u,
          (__int64)&v57,
          v53,
          v11,
          0,
          (__m128)0LL,
          *(double *)&v8,
          (__m128i)v9,
          v49,
          0,
          v45,
          (__int64)v42,
          v44);
  *((_DWORD *)v12 + 16) = v55 + 8;
LABEL_6:
  if ( v57 )
    sub_161E7C0((__int64)&v57, v57);
  return v12;
}

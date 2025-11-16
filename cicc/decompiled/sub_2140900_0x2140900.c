// Function: sub_2140900
// Address: 0x2140900
//
__int64 *__fastcall sub_2140900(__int64 a1, unsigned __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int *v6; // rax
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // r10
  __int64 v9; // r11
  __int64 v10; // rsi
  unsigned __int8 *v11; // rax
  unsigned __int64 v12; // r14
  unsigned int v13; // r15d
  unsigned int v14; // edx
  unsigned int v15; // ebx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int128 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rbx
  __int64 v23; // r15
  __int64 v24; // rsi
  unsigned __int8 *v25; // rax
  __int64 v26; // rcx
  unsigned int v27; // edx
  unsigned int v28; // ebx
  unsigned __int64 v29; // rdx
  __int64 v30; // r8
  __int64 v31; // r9
  __int128 v32; // rax
  __int64 *v33; // r14
  __int64 v34; // rdx
  __int64 v35; // r15
  __int64 v36; // rdx
  char v37; // cl
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r9
  __int64 *v41; // r10
  const void ***v42; // rcx
  int v43; // edx
  int v44; // r13d
  __int64 v45; // rsi
  const __m128i *v46; // r9
  __int64 *v47; // r13
  __int128 v49; // [rsp-20h] [rbp-D0h]
  __int64 v50; // [rsp+8h] [rbp-A8h]
  __int64 v51; // [rsp+10h] [rbp-A0h]
  __int64 v52; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v53; // [rsp+18h] [rbp-98h]
  __int64 *v54; // [rsp+18h] [rbp-98h]
  const void ***v55; // [rsp+18h] [rbp-98h]
  __int64 v56; // [rsp+20h] [rbp-90h]
  __int64 v57; // [rsp+20h] [rbp-90h]
  __int64 v58; // [rsp+28h] [rbp-88h]
  unsigned __int64 v59; // [rsp+30h] [rbp-80h]
  __int64 *v60; // [rsp+30h] [rbp-80h]
  __int128 v61; // [rsp+30h] [rbp-80h]
  __int64 v62; // [rsp+38h] [rbp-78h]
  __int64 v63; // [rsp+40h] [rbp-70h]
  __int64 *v64; // [rsp+40h] [rbp-70h]
  __int64 v65; // [rsp+40h] [rbp-70h]
  __int64 v66; // [rsp+50h] [rbp-60h] BYREF
  int v67; // [rsp+58h] [rbp-58h]
  __int64 v68; // [rsp+60h] [rbp-50h] BYREF
  __int64 v69; // [rsp+68h] [rbp-48h]
  char v70; // [rsp+70h] [rbp-40h]
  __int64 v71; // [rsp+78h] [rbp-38h]

  v6 = *(unsigned int **)(a2 + 32);
  v7 = *(_QWORD *)v6;
  v8 = *(_QWORD *)v6;
  v9 = *((_QWORD *)v6 + 1);
  v10 = *(_QWORD *)(*(_QWORD *)v6 + 72LL);
  v11 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v6 + 40LL) + 16LL * v6[2]);
  v12 = *((_QWORD *)v11 + 1);
  v13 = *v11;
  v68 = v10;
  if ( v10 )
  {
    v59 = v8;
    v62 = v9;
    sub_1623A60((__int64)&v68, v10, 2);
    v8 = v59;
    v9 = v62;
  }
  v58 = v9;
  LODWORD(v69) = *(_DWORD *)(v7 + 64);
  v63 = sub_2138AD0(a1, v8, v9);
  v15 = v14;
  v60 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v19 = sub_1D2EF30(v60, v13, v12, v16, v17, v18);
  v64 = sub_1D332F0(
          v60,
          148,
          (__int64)&v68,
          *(unsigned __int8 *)(*(_QWORD *)(v63 + 40) + 16LL * v15),
          *(const void ***)(*(_QWORD *)(v63 + 40) + 16LL * v15 + 8),
          0,
          a3,
          a4,
          a5,
          v63,
          v15 | v58 & 0xFFFFFFFF00000000LL,
          v19);
  v56 = v20;
  if ( v68 )
    sub_161E7C0((__int64)&v68, v68);
  *(_QWORD *)&v61 = v64;
  *((_QWORD *)&v61 + 1) = v56;
  v21 = *(_QWORD *)(a2 + 32);
  v22 = *(_QWORD *)(v21 + 40);
  v23 = *(_QWORD *)(v21 + 48);
  v24 = *(_QWORD *)(v22 + 72);
  v25 = (unsigned __int8 *)(*(_QWORD *)(v22 + 40) + 16LL * *(unsigned int *)(v21 + 48));
  v53 = *((_QWORD *)v25 + 1);
  v26 = *v25;
  v68 = v24;
  if ( v24 )
  {
    v51 = v26;
    sub_1623A60((__int64)&v68, v24, 2);
    v26 = v51;
  }
  v50 = v26;
  LODWORD(v69) = *(_DWORD *)(v22 + 64);
  v52 = sub_2138AD0(a1, v22, v23);
  v28 = v27;
  v29 = v53;
  v54 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v32 = sub_1D2EF30(v54, (unsigned int)v50, v29, v50, v30, v31);
  v33 = sub_1D332F0(
          v54,
          148,
          (__int64)&v68,
          *(unsigned __int8 *)(*(_QWORD *)(v52 + 40) + 16LL * v28),
          *(const void ***)(*(_QWORD *)(v52 + 40) + 16LL * v28 + 8),
          0,
          a3,
          a4,
          a5,
          v52,
          v28 | v23 & 0xFFFFFFFF00000000LL,
          v32);
  v35 = v34;
  if ( v68 )
    sub_161E7C0((__int64)&v68, v68);
  v36 = v64[5] + 16LL * (unsigned int)v56;
  v37 = *(_BYTE *)v36;
  v69 = *(_QWORD *)(v36 + 8);
  v38 = *(_QWORD *)(a2 + 40);
  LOBYTE(v68) = v37;
  LOBYTE(v36) = *(_BYTE *)(v38 + 16);
  v71 = *(_QWORD *)(v38 + 24);
  v70 = v36;
  v57 = *(_QWORD *)(a1 + 8);
  v65 = *(_QWORD *)(a2 + 32);
  v39 = sub_1D25C30(v57, (unsigned __int8 *)&v68, 2);
  v41 = (__int64 *)v57;
  v42 = (const void ***)v39;
  v44 = v43;
  v66 = *(_QWORD *)(a2 + 72);
  if ( v66 )
  {
    v55 = (const void ***)v39;
    sub_1623A60((__int64)&v66, v66, 2);
    v42 = v55;
    v41 = (__int64 *)v57;
  }
  v45 = *(unsigned __int16 *)(a2 + 24);
  v67 = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v49 + 1) = v35;
  *(_QWORD *)&v49 = v33;
  v47 = sub_1D37470(v41, v45, (__int64)&v66, v42, v44, v40, v61, v49, *(_OWORD *)(v65 + 80));
  if ( v66 )
    sub_161E7C0((__int64)&v66, v66);
  sub_2013400(a1, a2, 1, (__int64)v47, (__m128i *)1, v46);
  return v47;
}

// Function: sub_3786360
// Address: 0x3786360
//
__m128i *__fastcall sub_3786360(__int64 a1, __int64 a2, __m128i a3)
{
  __int16 *v5; // rax
  __int16 v6; // dx
  unsigned __int64 *v7; // rax
  __int64 v8; // rsi
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // r13
  int v12; // edx
  __int64 v13; // rdx
  __int64 v14; // r9
  unsigned __int16 *v15; // rax
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  _QWORD *v20; // r13
  __int64 v21; // rsi
  bool v22; // al
  bool v23; // r8
  unsigned int *v24; // rcx
  __int64 v25; // rax
  __int16 v26; // di
  __int64 v27; // rax
  char v28; // al
  _QWORD *v29; // r12
  __int128 v30; // rax
  __int64 v31; // r9
  unsigned __int8 *v32; // rax
  __m128i *v33; // r12
  bool v35; // al
  __int64 v36; // rdx
  __int64 v37; // r8
  unsigned int *v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rdi
  unsigned __int64 v41; // rdx
  unsigned int v42; // eax
  __int64 v43; // r12
  unsigned int v44; // r13d
  __int64 v45; // rdx
  _QWORD *v46; // rax
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // r13
  unsigned __int64 v49; // r12
  _QWORD *v50; // rdi
  __m128i *v51; // rax
  __int64 v52; // rdx
  unsigned __int8 *v53; // rax
  __int64 v54; // r12
  __int64 v55; // r14
  unsigned int v56; // edx
  unsigned __int64 v57; // r13
  __int128 v58; // [rsp-10h] [rbp-150h]
  __int128 v59; // [rsp-10h] [rbp-150h]
  unsigned int *v60; // [rsp+0h] [rbp-140h]
  char v61; // [rsp+8h] [rbp-138h]
  unsigned __int64 v62; // [rsp+10h] [rbp-130h]
  unsigned __int64 v63; // [rsp+18h] [rbp-128h]
  __int64 v64; // [rsp+20h] [rbp-120h]
  __int64 v65; // [rsp+20h] [rbp-120h]
  __int64 v66; // [rsp+20h] [rbp-120h]
  unsigned __int8 v67; // [rsp+20h] [rbp-120h]
  __int64 v68; // [rsp+20h] [rbp-120h]
  __int64 v69; // [rsp+28h] [rbp-118h]
  __int64 v70; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v71; // [rsp+58h] [rbp-E8h]
  __int64 v72; // [rsp+60h] [rbp-E0h] BYREF
  int v73; // [rsp+68h] [rbp-D8h]
  __int128 v74; // [rsp+70h] [rbp-D0h] BYREF
  __int128 v75; // [rsp+80h] [rbp-C0h] BYREF
  unsigned __int16 v76[4]; // [rsp+90h] [rbp-B0h] BYREF
  unsigned __int64 v77; // [rsp+98h] [rbp-A8h]
  unsigned __int64 v78; // [rsp+A0h] [rbp-A0h]
  __int64 v79; // [rsp+A8h] [rbp-98h]
  __int128 v80; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v81; // [rsp+C0h] [rbp-80h]
  __int128 v82; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v83; // [rsp+E0h] [rbp-60h]
  __int64 v84; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v85; // [rsp+F8h] [rbp-48h]
  __int64 v86; // [rsp+100h] [rbp-40h]
  __int64 v87; // [rsp+108h] [rbp-38h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v71 = *((_QWORD *)v5 + 1);
  v7 = *(unsigned __int64 **)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 80);
  LOWORD(v70) = v6;
  v9 = v7[5];
  v10 = v7[6];
  v72 = v8;
  v11 = v7[5];
  if ( v8 )
  {
    sub_B96E90((__int64)&v72, v8, 1);
    v7 = *(unsigned __int64 **)(a2 + 40);
  }
  v12 = *(_DWORD *)(a2 + 72);
  DWORD2(v74) = 0;
  DWORD2(v75) = 0;
  v73 = v12;
  v13 = v7[1];
  *(_QWORD *)&v74 = 0;
  *(_QWORD *)&v75 = 0;
  sub_375E8D0(a1, *v7, v13, (__int64)&v74, (__int64)&v75);
  v15 = (unsigned __int16 *)(*(_QWORD *)(v74 + 48) + 16LL * DWORD2(v74));
  v16 = *v15;
  v17 = *((_QWORD *)v15 + 1);
  LOWORD(v84) = v16;
  v85 = v17;
  if ( (_WORD)v16 )
    LODWORD(v18) = word_4456340[v16 - 1];
  else
    LODWORD(v18) = sub_3007240((__int64)&v84);
  v19 = *(_QWORD *)(v11 + 96);
  v18 = (unsigned int)v18;
  v20 = *(_QWORD **)(v19 + 24);
  if ( *(_DWORD *)(v19 + 32) > 0x40u )
    v20 = (_QWORD *)*v20;
  if ( (unsigned int)v18 > (unsigned __int64)v20 )
  {
    *((_QWORD *)&v58 + 1) = v10;
    *(_QWORD *)&v58 = v9;
    v32 = sub_3406EB0(*(_QWORD **)(a1 + 8), 0xA1u, (__int64)&v72, (unsigned int)v70, v71, v14, v74, v58);
    goto LABEL_22;
  }
  v21 = (unsigned __int16)v70;
  if ( (_WORD)v70 )
  {
    v23 = (unsigned __int16)(v70 - 176) <= 0x34u;
  }
  else
  {
    v64 = (unsigned int)v18;
    v22 = sub_3007100((__int64)&v70);
    v21 = (unsigned int)v21;
    v18 = v64;
    v23 = v22;
  }
  v24 = *(unsigned int **)(a2 + 40);
  v25 = *(_QWORD *)(*(_QWORD *)v24 + 48LL) + 16LL * v24[2];
  v26 = *(_WORD *)v25;
  v27 = *(_QWORD *)(v25 + 8);
  LOWORD(v84) = v26;
  v85 = v27;
  if ( v26 )
  {
    if ( (unsigned __int16)(v26 - 176) <= 0x34u != v23 )
      goto LABEL_12;
    goto LABEL_20;
  }
  v60 = v24;
  v61 = v23;
  v65 = v18;
  v28 = sub_3007100((__int64)&v84);
  v24 = v60;
  v21 = (unsigned int)v21;
  v18 = v65;
  if ( v28 == v61 )
  {
LABEL_20:
    v29 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)&v30 = sub_3400EE0((__int64)v29, (__int64)v20 - v18, (__int64)&v72, 0, a3);
    v32 = sub_3406EB0(v29, 0xA1u, (__int64)&v72, (unsigned int)v70, v71, v31, v75, v30);
LABEL_22:
    v33 = (__m128i *)v32;
    goto LABEL_23;
  }
LABEL_12:
  if ( (_WORD)v21 )
  {
    if ( (unsigned __int16)(v21 - 17) <= 0xD3u )
      LOWORD(v21) = word_4456580[(unsigned __int16)v21 - 1];
    goto LABEL_15;
  }
  v66 = (__int64)v24;
  v35 = sub_30070B0((__int64)&v70);
  v38 = (unsigned int *)v66;
  if ( v35 )
  {
    LOWORD(v21) = sub_3009970((__int64)&v70, v21, v36, v66, v37);
LABEL_15:
    if ( (_WORD)v21 == 2 )
      sub_C64ED0("Don't know how to extract fixed-width predicate subvector from a scalable predicate vector", 1u);
    v38 = *(unsigned int **)(a2 + 40);
  }
  v39 = *(_QWORD *)(*(_QWORD *)v38 + 48LL) + 16LL * v38[2];
  v40 = *(_QWORD *)(a1 + 8);
  v41 = *(_QWORD *)(v39 + 8);
  v62 = *(_QWORD *)v38;
  v63 = *((_QWORD *)v38 + 1);
  v76[0] = *(_WORD *)v39;
  v77 = v41;
  v42 = sub_33CD850(v40, *(unsigned int *)v76, v41, 0);
  v43 = *(_QWORD *)(a1 + 8);
  v44 = v42;
  v67 = v42;
  v84 = sub_2D5B750(v76);
  v85 = v45;
  LOBYTE(v79) = v45;
  v78 = (unsigned __int64)(v84 + 7) >> 3;
  v46 = sub_33EDE90(v43, v78, v79, v44);
  v48 = v47;
  v49 = (unsigned __int64)v46;
  sub_2EAC300((__int64)&v80, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 40LL), *((_DWORD *)v46 + 24), 0);
  v50 = *(_QWORD **)(a1 + 8);
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v51 = sub_33F4560(
          v50,
          (unsigned __int64)(v50 + 36),
          0,
          (__int64)&v72,
          v62,
          v63,
          v49,
          v48,
          v80,
          v81,
          v67,
          0,
          (__int64)&v84);
  *((_QWORD *)&v59 + 1) = v10;
  *(_QWORD *)&v59 = v9;
  v69 = v52;
  v68 = (__int64)v51;
  v53 = sub_3465D80(a3, *(_QWORD *)a1, *(_QWORD **)(a1 + 8), v49, v48, *(unsigned int *)v76, v77, v70, v71, v59);
  v84 = 0;
  v54 = (__int64)v53;
  v55 = *(_QWORD *)(a1 + 8);
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v57 = v56 | v48 & 0xFFFFFFFF00000000LL;
  sub_2EAC3A0((__int64)&v82, *(__int64 **)(v55 + 40));
  v33 = sub_33F1F00(
          (__int64 *)v55,
          (unsigned int)v70,
          v71,
          (__int64)&v72,
          v68,
          v69,
          v54,
          v57,
          v82,
          v83,
          0,
          0,
          (__int64)&v84,
          0);
LABEL_23:
  if ( v72 )
    sub_B91220((__int64)&v72, v72);
  return v33;
}

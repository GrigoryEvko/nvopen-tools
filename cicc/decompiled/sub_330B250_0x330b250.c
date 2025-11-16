// Function: sub_330B250
// Address: 0x330b250
//
__int64 __fastcall sub_330B250(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  unsigned __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // rdx
  int v12; // r15d
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rax
  int v17; // eax
  char v18; // al
  __int64 v19; // r9
  __int64 v20; // rdi
  __int64 v21; // rax
  int v22; // edx
  __int64 v23; // r13
  __int64 v25; // rsi
  unsigned int v26; // edx
  unsigned __int64 v27; // rdi
  unsigned int v28; // ecx
  const void **v29; // rsi
  int v30; // eax
  __int64 v31; // r12
  __int128 v32; // rax
  int v33; // r9d
  __int64 v34; // rdi
  __int64 v35; // rax
  int v36; // r9d
  int v37; // edx
  __int64 v38; // r13
  __int64 v39; // rax
  unsigned int v40; // edx
  __int64 v41; // r14
  int v42; // edx
  int v43; // r9d
  __int64 v44; // rax
  unsigned int v45; // edx
  __int64 v46; // rax
  __int64 v47; // r14
  int v48; // edx
  __int64 v49; // rax
  unsigned int v50; // edx
  int v51; // edx
  int v52; // r9d
  unsigned __int64 v53; // rdi
  unsigned __int64 v54; // rax
  __int64 v55; // [rsp+0h] [rbp-E0h]
  unsigned int v56; // [rsp+14h] [rbp-CCh]
  unsigned int v57; // [rsp+28h] [rbp-B8h]
  int v58; // [rsp+2Ch] [rbp-B4h]
  __int128 v59; // [rsp+30h] [rbp-B0h]
  int v60; // [rsp+40h] [rbp-A0h]
  __int64 v61; // [rsp+48h] [rbp-98h]
  unsigned __int16 v62; // [rsp+52h] [rbp-8Eh]
  unsigned int v63; // [rsp+54h] [rbp-8Ch]
  unsigned int v64; // [rsp+58h] [rbp-88h]
  int v65; // [rsp+58h] [rbp-88h]
  int v66; // [rsp+58h] [rbp-88h]
  __int128 v67; // [rsp+60h] [rbp-80h]
  int v68; // [rsp+60h] [rbp-80h]
  __int64 v69; // [rsp+70h] [rbp-70h] BYREF
  int v70; // [rsp+78h] [rbp-68h]
  unsigned __int64 v71; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v72; // [rsp+88h] [rbp-58h]
  unsigned __int64 v73; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v74; // [rsp+98h] [rbp-48h]
  __int64 v75; // [rsp+A0h] [rbp-40h]
  int v76; // [rsp+A8h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)v8;
  v10 = *(_QWORD *)(v8 + 40);
  v63 = *(_DWORD *)(v8 + 8);
  v59 = (__int128)_mm_loadu_si128((const __m128i *)v8);
  v64 = *(_DWORD *)(v8 + 48);
  v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v8 + 48LL) + 16LL * v63 + 8);
  v12 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v8 + 48LL) + 16LL * v63);
  v67 = (__int128)_mm_loadu_si128((const __m128i *)(v8 + 40));
  v60 = v11;
  v58 = *(_DWORD *)(a2 + 24);
  v13 = *(_QWORD *)(a2 + 48);
  v14 = *(_QWORD *)(a2 + 80);
  v15 = *(unsigned __int16 *)(v13 + 16);
  v16 = *(_QWORD *)(v13 + 24);
  v69 = v14;
  v62 = v15;
  v61 = v16;
  if ( v14 )
    sub_B96E90((__int64)&v69, v14, 1);
  v70 = *(_DWORD *)(a2 + 72);
  if ( !(unsigned __int8)sub_33CF8A0(a2, 1, v11, v15, a5, a6) )
  {
    v34 = *a1;
    v73 = 0;
    v74 = 0;
    v35 = sub_33F17F0(v34, 51, &v73, v62, v61);
    v65 = v37;
    v38 = v35;
    if ( v73 )
      sub_B91220((__int64)&v73, v73);
    v39 = sub_3406EB0(*a1, 57, (unsigned int)&v69, v12, v60, v36, v59, v67);
    v75 = v38;
    v73 = v39;
    v74 = v40;
    v76 = v65;
    goto LABEL_10;
  }
  if ( v64 == v63 && v9 == v10 )
  {
    v46 = sub_3400BD0(*a1, 0, (unsigned int)&v69, v62, v61, 0, 0);
    LODWORD(v55) = 0;
    v47 = v46;
    v68 = v48;
    v49 = sub_3400BD0(*a1, 0, (unsigned int)&v69, v12, v60, 0, v55);
    v75 = v47;
    v73 = v49;
    v74 = v50;
    v76 = v68;
    goto LABEL_10;
  }
  v17 = *(_DWORD *)(v10 + 24);
  if ( v17 != 35 && v17 != 11 || (*(_BYTE *)(v10 + 32) & 8) != 0 || v58 != 78 )
  {
LABEL_8:
    v18 = sub_33E0720(v67, *((_QWORD *)&v67 + 1), 0);
    v20 = *a1;
    if ( v18 )
    {
      v21 = sub_3400BD0(v20, 0, (unsigned int)&v69, v62, v61, 0, 0);
      v73 = v9;
      v75 = v21;
      v74 = v63;
      v76 = v22;
LABEL_10:
      v23 = sub_32EB790((__int64)a1, a2, (__int64 *)&v73, 2, 1);
      goto LABEL_11;
    }
    if ( v58 == 78 )
    {
      if ( (unsigned int)sub_33DF620(v20, v9, v63, v10, v64, v19) )
        goto LABEL_33;
    }
    else if ( (unsigned int)sub_33DD890(v20, v9, v63, v10, v64, v19) )
    {
      if ( !(unsigned __int8)sub_33E07E0(v59, *((_QWORD *)&v59 + 1), 0) )
      {
LABEL_33:
        v23 = 0;
        goto LABEL_11;
      }
      v41 = sub_3400BD0(*a1, 0, (unsigned int)&v69, v62, v61, 0, 0);
      v66 = v51;
      v44 = sub_3406EB0(*a1, 188, (unsigned int)&v69, v12, v60, v52, v67, v59);
LABEL_36:
      v75 = v41;
      v74 = v45;
      v73 = v44;
      v76 = v66;
      goto LABEL_10;
    }
    v41 = sub_3400BD0(*a1, 0, (unsigned int)&v69, v62, v61, 0, 0);
    v66 = v42;
    v44 = sub_3406EB0(*a1, 57, (unsigned int)&v69, v12, v60, v43, v59, v67);
    goto LABEL_36;
  }
  v25 = *(_QWORD *)(v10 + 96);
  v26 = *(_DWORD *)(v25 + 32);
  v27 = *(_QWORD *)(v25 + 24);
  v28 = v26 - 1;
  if ( v26 <= 0x40 )
  {
    if ( 1LL << v28 != v27 )
    {
      v72 = *(_DWORD *)(v25 + 32);
      v31 = *a1;
LABEL_41:
      v53 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v26) & ~v27;
      v54 = 0;
      if ( v26 )
        v54 = v53;
      v71 = v54;
      goto LABEL_21;
    }
    goto LABEL_8;
  }
  v29 = (const void **)(v25 + 24);
  v57 = v26 - 1;
  if ( (*(_QWORD *)(v27 + 8LL * (v28 >> 6)) & (1LL << v28)) != 0 )
  {
    v56 = v26;
    v30 = sub_C44590((__int64)v29);
    v26 = v56;
    if ( v30 == v57 )
      goto LABEL_8;
  }
  v72 = v26;
  v31 = *a1;
  sub_C43780((__int64)&v71, v29);
  v26 = v72;
  if ( v72 <= 0x40 )
  {
    v27 = v71;
    goto LABEL_41;
  }
  sub_C43D10((__int64)&v71);
LABEL_21:
  sub_C46250((__int64)&v71);
  v74 = v72;
  v72 = 0;
  v73 = v71;
  *(_QWORD *)&v32 = sub_34007B0(v31, (unsigned int)&v73, (unsigned int)&v69, v12, v60, 0, 0);
  v23 = sub_3411F20(v31, 76, (unsigned int)&v69, *(_QWORD *)(a2 + 48), *(_DWORD *)(a2 + 68), v33, v59, v32);
  if ( v74 > 0x40 && v73 )
    j_j___libc_free_0_0(v73);
  if ( v72 > 0x40 && v71 )
    j_j___libc_free_0_0(v71);
LABEL_11:
  if ( v69 )
    sub_B91220((__int64)&v69, v69);
  return v23;
}

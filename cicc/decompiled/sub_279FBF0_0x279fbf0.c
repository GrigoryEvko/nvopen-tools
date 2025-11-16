// Function: sub_279FBF0
// Address: 0x279fbf0
//
__int64 __fastcall sub_279FBF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v10; // r15
  __int64 v12; // rdx
  __int64 *v13; // rax
  unsigned __int64 *v14; // rsi
  unsigned int v15; // ebx
  __int64 v16; // r15
  __int64 v18; // rdi
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned int v28; // eax
  __int64 v29; // rsi
  unsigned int v30; // r14d
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 i; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  int v39; // eax
  _QWORD *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  _QWORD *v45; // rbx
  _QWORD *v46; // r15
  void (__fastcall *v47)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v48; // rax
  _QWORD *v49; // rbx
  _QWORD *v50; // r12
  __int64 v51; // rax
  _QWORD *v52; // rbx
  _QWORD *v53; // r12
  __int64 v54; // rax
  unsigned int v56; // eax
  _QWORD *v57; // rdi
  int v58; // eax
  int v59; // ebx
  unsigned __int64 v60; // rdx
  unsigned __int64 v61; // rax
  _QWORD *v62; // rax
  __int64 v63; // rdx
  unsigned int v64; // eax
  _QWORD *v65; // rax
  __int64 v66; // [rsp-10h] [rbp-660h]
  __int64 v67; // [rsp-8h] [rbp-658h]
  _QWORD v68[2]; // [rsp+30h] [rbp-620h] BYREF
  __int64 v69; // [rsp+40h] [rbp-610h]
  __int64 v70; // [rsp+48h] [rbp-608h]
  unsigned int v71; // [rsp+50h] [rbp-600h]
  unsigned __int64 v72[2]; // [rsp+60h] [rbp-5F0h] BYREF
  _BYTE v73[512]; // [rsp+70h] [rbp-5E0h] BYREF
  __int64 v74; // [rsp+270h] [rbp-3E0h]
  __int64 v75; // [rsp+278h] [rbp-3D8h]
  __int64 v76; // [rsp+280h] [rbp-3D0h]
  __int64 v77; // [rsp+288h] [rbp-3C8h]
  char v78; // [rsp+290h] [rbp-3C0h]
  __int64 v79; // [rsp+298h] [rbp-3B8h]
  char *v80; // [rsp+2A0h] [rbp-3B0h]
  __int64 v81; // [rsp+2A8h] [rbp-3A8h]
  int v82; // [rsp+2B0h] [rbp-3A0h]
  char v83; // [rsp+2B4h] [rbp-39Ch]
  char v84; // [rsp+2B8h] [rbp-398h] BYREF
  __int16 v85; // [rsp+2F8h] [rbp-358h]
  _QWORD *v86; // [rsp+300h] [rbp-350h]
  _QWORD *v87; // [rsp+308h] [rbp-348h]
  __int64 v88; // [rsp+310h] [rbp-340h]
  __int64 v89; // [rsp+320h] [rbp-330h] BYREF
  _BYTE *v90; // [rsp+328h] [rbp-328h]
  __int64 v91; // [rsp+330h] [rbp-320h]
  _BYTE v92[384]; // [rsp+338h] [rbp-318h] BYREF
  __int64 v93; // [rsp+4B8h] [rbp-198h]
  char *v94; // [rsp+4C0h] [rbp-190h]
  __int64 v95; // [rsp+4C8h] [rbp-188h]
  int v96; // [rsp+4D0h] [rbp-180h]
  char v97; // [rsp+4D4h] [rbp-17Ch]
  char v98; // [rsp+4D8h] [rbp-178h] BYREF
  _BYTE *v99; // [rsp+518h] [rbp-138h]
  __int64 v100; // [rsp+520h] [rbp-130h]
  _BYTE v101[200]; // [rsp+528h] [rbp-128h] BYREF
  int v102; // [rsp+5F0h] [rbp-60h] BYREF
  _QWORD *v103; // [rsp+5F8h] [rbp-58h]
  int *v104; // [rsp+600h] [rbp-50h]
  int *v105; // [rsp+608h] [rbp-48h]
  __int64 v106; // [rsp+610h] [rbp-40h]

  v10 = a1;
  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 24) = a4;
  *(_QWORD *)(a1 + 16) = a7;
  *(_QWORD *)(a1 + 328) = a7;
  v68[0] = &unk_4A32730;
  *(_QWORD *)(a1 + 96) = a9;
  *(_QWORD *)(a1 + 104) = v68;
  v12 = 0x1000000000LL;
  *(_QWORD *)(a1 + 112) = a8;
  v89 = a10;
  v13 = &v89;
  *(_QWORD *)(a1 + 336) = a4;
  *(_QWORD *)(a1 + 480) = a4;
  *(_QWORD *)(a1 + 32) = a5;
  *(_QWORD *)(a1 + 320) = a6;
  *(_BYTE *)(a1 + 128) = 1;
  v68[1] = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  *(_BYTE *)(a1 + 760) = 1;
  v90 = v92;
  v94 = &v98;
  v99 = v101;
  v104 = &v102;
  v105 = &v102;
  if ( !a10 )
    v13 = 0;
  v14 = 0;
  v91 = 0x1000000000LL;
  v93 = 0;
  *(_QWORD *)(a1 + 120) = v13;
  v72[0] = (unsigned __int64)v73;
  v95 = 8;
  v96 = 0;
  v97 = 1;
  v100 = 0x800000000LL;
  v102 = 0;
  v103 = 0;
  v106 = 0;
  v72[1] = 0x1000000000LL;
  v74 = 0;
  v75 = 0;
  v76 = a4;
  v77 = 0;
  v78 = 1;
  v79 = 0;
  v80 = &v84;
  v81 = 8;
  v82 = 0;
  v83 = 1;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  if ( (_BYTE)qword_4FFB5E8 )
  {
    v14 = (unsigned __int64 *)a2;
    sub_2794B40(a1, a2);
  }
  v15 = 0;
  if ( a2 + 72 != *(_QWORD *)(a2 + 80) )
  {
    v16 = *(_QWORD *)(a2 + 80);
    do
    {
      v18 = v16;
      v16 = *(_QWORD *)(v16 + 8);
      v14 = v72;
      v19 = sub_F39690(v18 - 24, (__int64)v72, a8, *(__int64 **)(a1 + 120), *(_QWORD *)(a1 + 16), 0, 0);
      v12 = v66;
      a4 = v67;
      v15 |= v19;
    }
    while ( a2 + 72 != v16 );
    v10 = a1;
  }
  sub_FFCE90((__int64)v72, (__int64)v14, v12, a4, a5, a6);
  sub_FFD870((__int64)v72, (__int64)v14, v20, v21, v22, v23);
  sub_FFBC40((__int64)v72, (__int64)v14);
  v28 = v15;
  do
  {
    v29 = a2;
    v30 = v28;
    v28 = sub_279FB50(v10, a2, v24, v25, v26, v27);
  }
  while ( (_BYTE)v28 );
  if ( (unsigned __int8)sub_278A900((unsigned __int8 *)v10) && (unsigned __int8)sub_278A920(v10) )
  {
    sub_2794A80(v10);
    v64 = v30;
    do
    {
      v29 = a2;
      v30 = v64;
      v64 = sub_2799320(v10, a2);
    }
    while ( (_BYTE)v64 );
  }
  sub_278F580(v10, v29, v31, v32, v33, v34);
  v39 = *(_DWORD *)(v10 + 64);
  ++*(_QWORD *)(v10 + 48);
  if ( !v39 )
  {
    if ( !*(_DWORD *)(v10 + 68) )
      goto LABEL_18;
    i = *(unsigned int *)(v10 + 72);
    if ( (unsigned int)i > 0x40 )
    {
      v29 = 8 * i;
      sub_C7D6A0(*(_QWORD *)(v10 + 56), 8 * i, 8);
      *(_QWORD *)(v10 + 56) = 0;
      *(_QWORD *)(v10 + 64) = 0;
      *(_DWORD *)(v10 + 72) = 0;
      goto LABEL_18;
    }
    goto LABEL_15;
  }
  v36 = (unsigned int)(4 * v39);
  v29 = 64;
  i = *(unsigned int *)(v10 + 72);
  if ( (unsigned int)v36 < 0x40 )
    v36 = 64;
  if ( (unsigned int)i <= (unsigned int)v36 )
  {
LABEL_15:
    v40 = *(_QWORD **)(v10 + 56);
    for ( i = (__int64)&v40[i]; (_QWORD *)i != v40; ++v40 )
      *v40 = -4096;
    *(_QWORD *)(v10 + 64) = 0;
    goto LABEL_18;
  }
  v56 = v39 - 1;
  if ( !v56 )
  {
    v57 = *(_QWORD **)(v10 + 56);
    v59 = 64;
LABEL_61:
    sub_C7D6A0((__int64)v57, 8 * i, 8);
    v29 = 8;
    v60 = ((((((((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
             | (4 * v59 / 3u + 1)
             | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 4)
           | (((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
           | (4 * v59 / 3u + 1)
           | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
           | (4 * v59 / 3u + 1)
           | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 4)
         | (((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
         | (4 * v59 / 3u + 1)
         | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 16;
    v61 = (v60
         | (((((((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
             | (4 * v59 / 3u + 1)
             | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 4)
           | (((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
           | (4 * v59 / 3u + 1)
           | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
           | (4 * v59 / 3u + 1)
           | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 4)
         | (((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
         | (4 * v59 / 3u + 1)
         | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(v10 + 72) = v61;
    v62 = (_QWORD *)sub_C7D670(8 * v61, 8);
    v63 = *(unsigned int *)(v10 + 72);
    *(_QWORD *)(v10 + 64) = 0;
    *(_QWORD *)(v10 + 56) = v62;
    for ( i = (__int64)&v62[v63]; (_QWORD *)i != v62; ++v62 )
    {
      if ( v62 )
        *v62 = -4096;
    }
    goto LABEL_18;
  }
  _BitScanReverse(&v56, v56);
  v57 = *(_QWORD **)(v10 + 56);
  v58 = v56 ^ 0x1F;
  v36 = (unsigned int)(33 - v58);
  v59 = 1 << (33 - v58);
  if ( v59 < 64 )
    v59 = 64;
  if ( (_DWORD)i != v59 )
    goto LABEL_61;
  *(_QWORD *)(v10 + 64) = 0;
  v65 = &v57[i];
  do
  {
    if ( v57 )
      *v57 = -4096;
    ++v57;
  }
  while ( v65 != v57 );
LABEL_18:
  *(_DWORD *)(v10 + 88) = 0;
  if ( a10 && byte_4F8F8E8[0] )
  {
    v29 = 0;
    nullsub_390();
  }
  sub_FFCE90((__int64)v72, v29, i, v36, v37, v38);
  sub_FFD870((__int64)v72, v29, v41, v42, v43, v44);
  sub_FFBC40((__int64)v72, v29);
  v45 = v87;
  v46 = v86;
  if ( v87 != v86 )
  {
    do
    {
      v47 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v46[7];
      *v46 = &unk_49E5048;
      if ( v47 )
        v47(v46 + 5, v46 + 5, 3);
      *v46 = &unk_49DB368;
      v48 = v46[3];
      if ( v48 != 0 && v48 != -4096 && v48 != -8192 )
        sub_BD60C0(v46 + 1);
      v46 += 9;
    }
    while ( v45 != v46 );
    v46 = v86;
  }
  if ( v46 )
    j_j___libc_free_0((unsigned __int64)v46);
  if ( !v83 )
    _libc_free((unsigned __int64)v80);
  if ( (_BYTE *)v72[0] != v73 )
    _libc_free(v72[0]);
  sub_2789AD0(v103);
  v49 = v99;
  v50 = &v99[24 * (unsigned int)v100];
  if ( v99 != (_BYTE *)v50 )
  {
    do
    {
      v51 = *(v50 - 1);
      v50 -= 3;
      if ( v51 != -4096 && v51 != 0 && v51 != -8192 )
        sub_BD60C0(v50);
    }
    while ( v49 != v50 );
    v50 = v99;
  }
  if ( v50 != (_QWORD *)v101 )
    _libc_free((unsigned __int64)v50);
  if ( !v97 )
    _libc_free((unsigned __int64)v94);
  v52 = v90;
  v53 = &v90[24 * (unsigned int)v91];
  if ( v90 != (_BYTE *)v53 )
  {
    do
    {
      v54 = *(v53 - 1);
      v53 -= 3;
      if ( v54 != -4096 && v54 != 0 && v54 != -8192 )
        sub_BD60C0(v53);
    }
    while ( v52 != v53 );
    v53 = v90;
  }
  if ( v53 != (_QWORD *)v92 )
    _libc_free((unsigned __int64)v53);
  v68[0] = &unk_4A20C88;
  sub_C7D6A0(v69, 16LL * v71, 8);
  return v30;
}

// Function: sub_2604F70
// Address: 0x2604f70
//
__int64 __fastcall sub_2604F70(_QWORD *a1, __int64 *a2)
{
  __int64 v4; // r14
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rdi
  unsigned __int8 v12; // al
  __int64 v13; // r8
  __int64 v14; // rdx
  unsigned __int64 v15; // r9
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // r8
  unsigned __int64 v19; // r9
  __int64 v20; // rdx
  char v21; // r15
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rbx
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  unsigned __int64 *v27; // rcx
  __int64 v28; // rdx
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rax
  unsigned __int64 *v31; // rcx
  unsigned __int64 v32; // rsi
  __int64 v33; // rdx
  _QWORD *v34; // rsi
  unsigned __int64 v35; // rcx
  _QWORD *v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // r15
  __int64 v39; // rax
  char v40; // al
  __int64 v41; // r8
  unsigned int v42; // r14d
  __int64 v43; // rax
  __int64 v44; // rbx
  __int64 v45; // r15
  __int64 v47; // r15
  _QWORD *v48; // rax
  __int64 v49; // [rsp+8h] [rbp-1B8h]
  __int64 v50; // [rsp+10h] [rbp-1B0h]
  unsigned int v51; // [rsp+10h] [rbp-1B0h]
  __int64 v52; // [rsp+20h] [rbp-1A0h]
  __int64 v53; // [rsp+20h] [rbp-1A0h]
  __int64 v54; // [rsp+30h] [rbp-190h] BYREF
  __int64 v55; // [rsp+38h] [rbp-188h]
  __int64 v56; // [rsp+40h] [rbp-180h]
  __int64 v57; // [rsp+48h] [rbp-178h]
  __int64 *v58; // [rsp+50h] [rbp-170h]
  __int64 v59; // [rsp+58h] [rbp-168h]
  __int64 v60; // [rsp+60h] [rbp-160h] BYREF
  __int64 v61; // [rsp+68h] [rbp-158h]
  __int64 v62; // [rsp+70h] [rbp-150h]
  __int64 v63; // [rsp+78h] [rbp-148h]
  _QWORD *v64; // [rsp+80h] [rbp-140h]
  __int64 v65; // [rsp+88h] [rbp-138h]
  _QWORD v66[6]; // [rsp+90h] [rbp-130h] BYREF
  unsigned __int64 v67[2]; // [rsp+C0h] [rbp-100h] BYREF
  char v68; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v69; // [rsp+158h] [rbp-68h]
  unsigned int v70; // [rsp+168h] [rbp-58h]
  __int64 v71; // [rsp+178h] [rbp-48h]
  unsigned int v72; // [rsp+188h] [rbp-38h]

  v4 = a2[34];
  v64 = v66;
  v5 = *(_QWORD *)(v4 + 72);
  v58 = &v60;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v65 = 0;
  memset(v66, 0, 32);
  v66[4] = v67;
  v66[5] = 0;
  sub_29B4290(v67, v5);
  v6 = sub_29B4E00(a2[29], v67, &v54, &v60);
  a2[31] = v6;
  if ( v6 )
  {
    v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL) + 40LL);
    v8 = sub_AA54C0(v7);
    a2[33] = v8;
    if ( v4 == v8 )
    {
      v47 = sub_AA54C0(v4);
      v48 = (_QWORD *)sub_986580(v47);
      sub_B43D60(v48);
      sub_AA80F0(v47, (unsigned __int64 *)(v47 + 48), 0, v4, *(__int64 **)(v4 + 56), 1, (__int64 *)(v4 + 48), 0);
      a2[33] = v47;
      sub_AA5450((_QWORD *)v4);
    }
    v9 = *a2;
    a2[34] = v7;
    a2[35] = v7;
    v10 = *(_QWORD *)(v7 + 56);
    v11 = (__int64)(a1 + 51);
    v52 = *(_QWORD *)(*(_QWORD *)(v9 + 8) + 160LL);
    if ( v10 )
      v10 -= 24;
    v12 = sub_25FE580(v11, v10);
    v14 = a1[39];
    a1[49] += 168LL;
    v15 = v12;
    v49 = (__int64)(a1 + 39);
    v16 = (v14 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[40] >= (unsigned __int64)(v16 + 168) && v14 )
    {
      a1[39] = v16 + 168;
    }
    else
    {
      v51 = v15;
      v16 = sub_9D1E70(v49, 168, 168, 3);
      v15 = v51;
    }
    v50 = v16;
    sub_22AF450(v16, v10, v15, v52, v13, v15);
    a2[1] = v50;
    v17 = sub_25FE580((__int64)(a1 + 51), v10);
    v20 = a1[39];
    a1[49] += 168LL;
    v21 = v17;
    v22 = (v20 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[40] >= (unsigned __int64)(v22 + 168) && v20 )
      a1[39] = v22 + 168;
    else
      v22 = sub_9D1E70(v49, 168, 168, 3);
    v23 = v52;
    v53 = v22;
    v24 = v7 + 48;
    sub_22AF450(v22, v10, v21, v23, v18, v19);
    v25 = *a2;
    a2[2] = v53;
    v26 = a2[1];
    v27 = *(unsigned __int64 **)(v25 + 8);
    v28 = *(_QWORD *)v26;
    v29 = *v27;
    *(_QWORD *)(v26 + 8) = v27;
    v29 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v26 = v29 | v28 & 7;
    *(_QWORD *)(v29 + 8) = v26;
    *v27 = *v27 & 7 | v26;
    v30 = a2[2];
    v31 = *(unsigned __int64 **)(*(_QWORD *)(*a2 + 16) + 8LL);
    v32 = *v31;
    v33 = *(_QWORD *)v30 & 7LL;
    *(_QWORD *)(v30 + 8) = v31;
    v32 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v30 = v32 | v33;
    *(_QWORD *)(v32 + 8) = v30;
    *v31 = *v31 & 7 | v30;
    v34 = *(_QWORD **)(*a2 + 8);
    v35 = *v34 & 0xFFFFFFFFFFFFFFF8LL;
    v36 = (_QWORD *)(**(_QWORD **)(*(_QWORD *)(*a2 + 16) + 8LL) & 0xFFFFFFFFFFFFFFF8LL);
    v37 = *v36 & 0xFFFFFFFFFFFFFFF8LL;
    *v36 = v35 | *v36 & 7LL;
    *(_QWORD *)(v35 + 8) = v36;
    *v34 &= 7uLL;
    *(_QWORD *)(v37 + 8) = 0;
    v38 = *(_QWORD *)(v24 + 8);
    if ( v24 == v38 )
    {
LABEL_25:
      v42 = 1;
      sub_2600BA0(a2);
      goto LABEL_26;
    }
    while ( 1 )
    {
      if ( !v38 )
        BUG();
      v40 = *(_BYTE *)(v38 - 24);
      v41 = v38 - 24;
      if ( v40 == 85 )
        break;
      if ( v40 == 61 )
      {
        sub_2604CE0((__int64)a1, (__int64)a2, (__int64)v64, (unsigned int)v65, v41);
        v38 = *(_QWORD *)(v38 + 8);
        if ( v24 == v38 )
          goto LABEL_25;
      }
      else
      {
LABEL_20:
        v38 = *(_QWORD *)(v38 + 8);
        if ( v24 == v38 )
          goto LABEL_25;
      }
    }
    v39 = *(_QWORD *)(v38 - 56);
    if ( v39 )
    {
      if ( *(_BYTE *)v39 )
      {
        v39 = 0;
      }
      else if ( *(_QWORD *)(v39 + 24) != *(_QWORD *)(v38 + 56) )
      {
        v39 = 0;
      }
    }
    if ( a2[31] == v39 )
      a2[30] = v41;
    goto LABEL_20;
  }
  v42 = 0;
  sub_2600BA0(a2);
LABEL_26:
  sub_C7D6A0(v71, 8LL * v72, 8);
  v43 = v70;
  if ( v70 )
  {
    v44 = v69;
    v45 = v69 + 40LL * v70;
    do
    {
      if ( *(_QWORD *)v44 != -8192 && *(_QWORD *)v44 != -4096 )
        sub_C7D6A0(*(_QWORD *)(v44 + 16), 8LL * *(unsigned int *)(v44 + 32), 8);
      v44 += 40;
    }
    while ( v45 != v44 );
    v43 = v70;
  }
  sub_C7D6A0(v69, 40 * v43, 8);
  if ( (char *)v67[0] != &v68 )
    _libc_free(v67[0]);
  sub_C7D6A0(0, 0, 8);
  if ( v64 != v66 )
    _libc_free((unsigned __int64)v64);
  sub_C7D6A0(v61, 8LL * (unsigned int)v63, 8);
  if ( v58 != &v60 )
    _libc_free((unsigned __int64)v58);
  sub_C7D6A0(v55, 8LL * (unsigned int)v57, 8);
  return v42;
}

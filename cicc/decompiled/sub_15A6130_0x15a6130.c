// Function: sub_15A6130
// Address: 0x15a6130
//
void __fastcall sub_15A6130(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rax
  int v4; // edx
  _QWORD *v5; // rax
  __int64 v6; // r12
  __int64 v7; // r15
  _BYTE *v8; // rdx
  __int64 v9; // r12
  char v10; // dl
  __int64 v11; // rsi
  _QWORD *v12; // rdi
  _QWORD *v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // r14
  __int64 v23; // rax
  __int64 *i; // r12
  __int64 *v25; // r12
  __int64 *v26; // r14
  unsigned int v27; // eax
  unsigned int v28; // eax
  __int64 *v29; // r15
  __int64 *j; // r14
  __int64 v31; // rdi
  __int64 v32; // rsi
  __int64 v33; // r12
  __int64 v34; // rdx
  int v35; // eax
  __int64 v36; // rax
  __int64 *v37; // r12
  __int64 *v38; // r14
  __int64 v39; // rdi
  __int64 v40; // r14
  __int64 v41; // r12
  __int64 v42; // rsi
  unsigned __int64 v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r12
  __int64 v48; // rax
  unsigned __int64 v49; // r9
  __int64 v50; // r15
  _QWORD *v51; // r12
  _QWORD *v52; // rax
  __int64 v53; // r8
  _QWORD *v54; // rcx
  __int64 v55; // rdi
  __int64 v56; // rax
  __int64 v57; // r12
  __int64 v58; // rax
  __int64 v59; // r12
  __int64 v60; // rax
  __int64 v61; // [rsp-10h] [rbp-220h]
  __int64 v62; // [rsp-8h] [rbp-218h]
  __int64 v63; // [rsp+0h] [rbp-210h]
  int v64; // [rsp+8h] [rbp-208h]
  __int64 v65; // [rsp+8h] [rbp-208h]
  __int64 *v66; // [rsp+10h] [rbp-200h] BYREF
  __int64 v67; // [rsp+18h] [rbp-1F8h]
  _BYTE v68[128]; // [rsp+20h] [rbp-1F0h] BYREF
  _BYTE *v69; // [rsp+A0h] [rbp-170h] BYREF
  __int64 v70; // [rsp+A8h] [rbp-168h]
  _BYTE v71[128]; // [rsp+B0h] [rbp-160h] BYREF
  __int64 v72; // [rsp+130h] [rbp-E0h] BYREF
  _BYTE *v73; // [rsp+138h] [rbp-D8h]
  _BYTE *v74; // [rsp+140h] [rbp-D0h]
  __int64 v75; // [rsp+148h] [rbp-C8h]
  int v76; // [rsp+150h] [rbp-C0h]
  _BYTE v77[184]; // [rsp+158h] [rbp-B8h] BYREF

  v1 = *(_QWORD *)(a1 + 16);
  if ( !v1 )
    return;
  v3 = sub_1627350(*(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 48), *(unsigned int *)(a1 + 56), 0, 1);
  sub_1630830(v1, 4, v3);
  v4 = *(_DWORD *)(a1 + 104);
  v66 = (__int64 *)v68;
  v67 = 0x1000000000LL;
  v5 = v77;
  v72 = 0;
  v73 = v77;
  v74 = v77;
  v75 = 16;
  v76 = 0;
  if ( !v4 )
    goto LABEL_21;
  v6 = (unsigned int)(v4 - 1);
  v7 = 0;
  v8 = v77;
  v9 = 8 * v6;
  while ( 1 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + v7);
    if ( v8 == (_BYTE *)v5 )
    {
      v12 = &v5[HIDWORD(v75)];
      if ( v12 != v5 )
      {
        v13 = 0;
        while ( v11 != *v5 )
        {
          if ( *v5 == -2 )
            v13 = v5;
          if ( v12 == ++v5 )
          {
            if ( !v13 )
              goto LABEL_65;
            *v13 = v11;
            --v76;
            ++v72;
            goto LABEL_16;
          }
        }
        goto LABEL_5;
      }
LABEL_65:
      if ( HIDWORD(v75) < (unsigned int)v75 )
        break;
    }
    sub_16CCBA0(&v72, v11);
    if ( v10 )
      goto LABEL_16;
LABEL_5:
    if ( v9 == v7 )
      goto LABEL_19;
LABEL_6:
    v8 = v74;
    v5 = v73;
    v7 += 8;
  }
  ++HIDWORD(v75);
  *v12 = v11;
  ++v72;
LABEL_16:
  v14 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + v7);
  v15 = (unsigned int)v67;
  if ( (unsigned int)v67 >= HIDWORD(v67) )
  {
    v65 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + v7);
    sub_16CD150(&v66, v68, 0, 8);
    v15 = (unsigned int)v67;
    v14 = v65;
  }
  v66[v15] = v14;
  LODWORD(v67) = v67 + 1;
  if ( v9 != v7 )
    goto LABEL_6;
LABEL_19:
  if ( (_DWORD)v67 )
  {
    v59 = *(_QWORD *)(a1 + 16);
    v60 = sub_1627350(*(_QWORD *)(a1 + 8), v66, (unsigned int)v67, 0, 1);
    sub_1630830(v59, 5, v60);
  }
LABEL_21:
  v16 = *(_QWORD *)(a1 + 144);
  v17 = sub_1627350(*(_QWORD *)(a1 + 8), v16, *(unsigned int *)(a1 + 152), 0, 1);
  v22 = (__int64 *)v17;
  if ( v17 )
  {
    v23 = 8LL * *(unsigned int *)(v17 + 8);
    for ( i = &v22[v23 / 0xFFFFFFFFFFFFFFF8LL]; v22 != i; ++i )
    {
      v16 = *i;
      sub_15A5DE0(a1, v16);
    }
  }
  v25 = v66;
  v26 = &v66[(unsigned int)v67];
  if ( v66 != v26 )
  {
    do
    {
      v16 = *v25;
      if ( *(_BYTE *)*v25 == 17 )
        sub_15A5DE0(a1, v16);
      ++v25;
    }
    while ( v26 != v25 );
  }
  v27 = *(_DWORD *)(a1 + 200);
  if ( v27 )
  {
    v57 = *(_QWORD *)(a1 + 16);
    v58 = sub_1627350(*(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 192), v27, 0, 1);
    v16 = 6;
    sub_1630830(v57, 6, v58);
    v28 = *(_DWORD *)(a1 + 248);
    if ( v28 )
      goto LABEL_55;
  }
  else
  {
    v28 = *(_DWORD *)(a1 + 248);
    if ( !v28 )
      goto LABEL_30;
LABEL_55:
    v49 = v28;
    v50 = *(_QWORD *)(a1 + 16);
    v70 = 0x1000000000LL;
    v51 = *(_QWORD **)(a1 + 240);
    v52 = v71;
    v53 = 8 * v49;
    v69 = v71;
    if ( v49 > 0x10 )
    {
      v63 = 8 * v49;
      v64 = v49;
      sub_16CD150(&v69, v71, v49, 8);
      v53 = v63;
      LODWORD(v49) = v64;
      v52 = &v69[8 * (unsigned int)v70];
    }
    v54 = (_QWORD *)((char *)v52 + v53);
    do
    {
      if ( v52 )
        *v52 = *v51;
      ++v52;
      ++v51;
    }
    while ( v54 != v52 );
    v55 = *(_QWORD *)(a1 + 8);
    LODWORD(v70) = v70 + v49;
    v56 = sub_1627350(v55, v69, (unsigned int)v70, 0, 1);
    v16 = 7;
    sub_1630830(v50, 7, v56);
    if ( v69 != v71 )
      _libc_free((unsigned __int64)v69);
  }
LABEL_30:
  v29 = *(__int64 **)(a1 + 328);
  for ( j = *(__int64 **)(a1 + 320); v29 != j; j += 8 )
  {
    while ( 1 )
    {
      v32 = j[5];
      v33 = *j;
      v34 = (j[6] - v32) >> 3;
      if ( *j )
        break;
      v47 = *(_QWORD *)(a1 + 16);
      v48 = sub_1627350(*(_QWORD *)(a1 + 8), v32, v34, 0, 1);
      v16 = 8;
      sub_1630830(v47, 8, v48);
LABEL_35:
      j += 8;
      if ( v29 == j )
        goto LABEL_39;
    }
    v35 = sub_15A6110(a1, v32, v34);
    v16 = 3;
    v36 = sub_15C6E80(
            *(_QWORD *)(a1 + 8),
            3,
            *(_DWORD *)(v33 + 24),
            *(_QWORD *)(v33 - 8LL * *(unsigned int *)(v33 + 8)),
            v35,
            0,
            1);
    v44 = v61;
    v45 = v62;
    if ( v33 != v36 )
    {
      v31 = *(_QWORD *)(v33 + 16);
      if ( (v31 & 4) != 0 )
      {
        v16 = v36;
        sub_16302D0(v31 & 0xFFFFFFFFFFFFFFF8LL, v36);
      }
      sub_16307F0(v33, v16, v44, v45, v46);
      goto LABEL_35;
    }
    sub_1630860(v33, 3, v61);
  }
LABEL_39:
  v37 = *(__int64 **)(a1 + 344);
  v38 = &v37[*(unsigned int *)(a1 + 352)];
  if ( v37 != v38 )
  {
    do
    {
      v39 = *v37;
      if ( *v37 && (*(_BYTE *)(v39 + 1) == 2 || *(_DWORD *)(v39 + 12)) )
        sub_161F200(v39, v16, v18, v19, v20, v21);
      ++v37;
    }
    while ( v38 != v37 );
    v40 = *(_QWORD *)(a1 + 344);
    v41 = v40 + 8LL * *(unsigned int *)(a1 + 352);
    while ( v40 != v41 )
    {
      while ( 1 )
      {
        v42 = *(_QWORD *)(v41 - 8);
        v41 -= 8;
        if ( !v42 )
          break;
        sub_161E7C0(v41);
        if ( v40 == v41 )
          goto LABEL_49;
      }
    }
  }
LABEL_49:
  *(_BYTE *)(a1 + 392) = 0;
  v43 = (unsigned __int64)v74;
  *(_DWORD *)(a1 + 352) = 0;
  if ( (_BYTE *)v43 != v73 )
    _libc_free(v43);
  if ( v66 != (__int64 *)v68 )
    _libc_free((unsigned __int64)v66);
}

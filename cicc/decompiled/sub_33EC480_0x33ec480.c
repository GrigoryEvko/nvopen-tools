// Function: sub_33EC480
// Address: 0x33ec480
//
__int64 __fastcall sub_33EC480(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned __int64 a4,
        int a5,
        __int64 a6,
        unsigned __int64 *a7,
        __int64 a8)
{
  __int64 v9; // r12
  bool v11; // zf
  _QWORD *v12; // rax
  __int64 v13; // r10
  __int64 v14; // rsi
  unsigned __int64 v15; // rdi
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 *v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // r13
  __int64 **v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rax
  _QWORD *v26; // rax
  unsigned __int64 v27; // r13
  unsigned int v28; // eax
  unsigned __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 *v33; // rdx
  _QWORD *v34; // rax
  _QWORD *v35; // r15
  __int64 v36; // r13
  _QWORD *v37; // r14
  _BYTE **v38; // r8
  _QWORD *v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  unsigned __int64 v42; // r10
  unsigned int v43; // r8d
  _QWORD *v44; // rax
  _QWORD *v45; // rdx
  _QWORD *v46; // [rsp+8h] [rbp-198h]
  __int64 v47; // [rsp+10h] [rbp-190h]
  unsigned int v48; // [rsp+18h] [rbp-188h]
  __int64 **v50; // [rsp+20h] [rbp-180h]
  _BYTE **v51; // [rsp+20h] [rbp-180h]
  unsigned __int64 v52; // [rsp+20h] [rbp-180h]
  __int64 *v53; // [rsp+38h] [rbp-168h] BYREF
  _BYTE *v54; // [rsp+40h] [rbp-160h] BYREF
  __int64 v55; // [rsp+48h] [rbp-158h]
  _BYTE v56[128]; // [rsp+50h] [rbp-150h] BYREF
  __int64 *v57; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v58; // [rsp+D8h] [rbp-C8h]
  __int64 v59; // [rsp+E0h] [rbp-C0h] BYREF
  int v60; // [rsp+E8h] [rbp-B8h]
  char v61; // [rsp+ECh] [rbp-B4h]
  char v62; // [rsp+F0h] [rbp-B0h] BYREF

  v9 = a2;
  v11 = *(_WORD *)(a4 + 16LL * (unsigned int)(a5 - 1)) == 262;
  v53 = 0;
  if ( !v11 )
  {
    v57 = &v59;
    v58 = 0x2000000000LL;
    sub_33C9670((__int64)&v57, a3, a4, a7, a8, a6);
    v54 = *(_BYTE **)(a2 + 80);
    if ( v54 )
      sub_B96E90((__int64)&v54, (__int64)v54, 1);
    LODWORD(v55) = *(_DWORD *)(a2 + 72);
    v12 = sub_33CCCF0(a1, (__int64)&v57, (__int64)&v54, (__int64 *)&v53);
    v13 = (__int64)v12;
    if ( v54 )
    {
      v46 = v12;
      sub_B91220((__int64)&v54, (__int64)v54);
      v13 = (__int64)v46;
    }
    if ( v13 )
    {
      v14 = *(_QWORD *)(a2 + 80);
      v54 = (_BYTE *)v14;
      if ( v14 )
      {
        v47 = v13;
        sub_B96E90((__int64)&v54, v14, 1);
        v13 = v47;
      }
      LODWORD(v55) = *(_DWORD *)(v9 + 72);
      v9 = sub_33CEC90(a1, v13, (__int64)&v54);
      if ( v54 )
        sub_B91220((__int64)&v54, (__int64)v54);
      v15 = (unsigned __int64)v57;
      if ( v57 != &v59 )
LABEL_45:
        _libc_free(v15);
      return v9;
    }
    if ( v57 != &v59 )
      _libc_free((unsigned __int64)v57);
  }
  if ( !(unsigned __int8)sub_33EB970(a1, a2, a3) )
    v53 = 0;
  *(_QWORD *)(a2 + 48) = a4;
  v19 = *(__int64 **)(a2 + 40);
  v58 = (__int64)&v62;
  v20 = *(unsigned int *)(a2 + 64);
  v21 = a3;
  *(_DWORD *)(a2 + 68) = a5;
  v57 = 0;
  v22 = &v19[5 * v20];
  *(_DWORD *)(a2 + 24) = a3;
  v59 = 16;
  v60 = 0;
  v61 = 1;
  if ( v22 == v19 )
    goto LABEL_31;
  v23 = &v57;
  do
  {
    while ( 1 )
    {
      v24 = *v19;
      v19 += 5;
      if ( v24 )
      {
        v21 = *(v19 - 2);
        v25 = *(v19 - 1);
        *(_QWORD *)v21 = v25;
        if ( v25 )
        {
          v21 = *(v19 - 2);
          *(_QWORD *)(v25 + 24) = v21;
        }
      }
      *(v19 - 5) = 0;
      *((_DWORD *)v19 - 8) = 0;
      if ( *(_QWORD *)(v24 + 56) )
        goto LABEL_19;
      if ( v61 )
      {
        v26 = (_QWORD *)v58;
        v21 = v58 + 8LL * HIDWORD(v59);
        if ( v58 != v21 )
        {
          while ( v24 != *v26 )
          {
            if ( (_QWORD *)v21 == ++v26 )
              goto LABEL_28;
          }
          goto LABEL_19;
        }
LABEL_28:
        if ( HIDWORD(v59) < (unsigned int)v59 )
          break;
      }
      v50 = v23;
      sub_C8CC70((__int64)v23, v24, v21, (__int64)v23, v17, v18);
      v23 = v50;
LABEL_19:
      if ( v19 == v22 )
        goto LABEL_30;
    }
    ++HIDWORD(v59);
    *(_QWORD *)v21 = v24;
    v57 = (__int64 *)((char *)v57 + 1);
  }
  while ( v19 != v22 );
LABEL_30:
  LODWORD(v21) = *(_DWORD *)(v9 + 24);
  v19 = *(__int64 **)(v9 + 40);
LABEL_31:
  if ( (int)v21 < 0 )
  {
    *(_QWORD *)(v9 + 96) = 0;
    *(_DWORD *)(v9 + 104) = 0;
  }
  if ( v19 )
  {
    v27 = *(unsigned int *)(v9 + 64);
    v28 = 0;
    if ( *(_DWORD *)(v9 + 64) )
    {
      if ( --v27 )
      {
        _BitScanReverse64(&v27, v27);
        v28 = 64 - (v27 ^ 0x3F);
        v27 = 8LL * (int)v28;
      }
    }
    v29 = *(unsigned int *)(a1 + 648);
    if ( (unsigned int)v29 > v28 || (v42 = v28 + 1, v43 = v28 + 1, v42 == v29) )
    {
      v30 = *(_QWORD *)(a1 + 640);
    }
    else if ( v42 < v29 )
    {
      *(_DWORD *)(a1 + 648) = v42;
      v30 = *(_QWORD *)(a1 + 640);
    }
    else
    {
      if ( v42 > *(unsigned int *)(a1 + 652) )
      {
        v48 = v28 + 1;
        v52 = v28 + 1;
        sub_C8D5F0(a1 + 640, (const void *)(a1 + 656), v42, 8u, v42, v18);
        v43 = v48;
        v42 = v52;
        v29 = *(unsigned int *)(a1 + 648);
      }
      v30 = *(_QWORD *)(a1 + 640);
      v44 = (_QWORD *)(v30 + 8 * v29);
      v45 = (_QWORD *)(v30 + 8 * v42);
      if ( v44 != v45 )
      {
        do
        {
          if ( v44 )
            *v44 = 0;
          ++v44;
        }
        while ( v45 != v44 );
        v30 = *(_QWORD *)(a1 + 640);
      }
      *(_DWORD *)(a1 + 648) = v43;
    }
    *v19 = *(_QWORD *)(v30 + v27);
    *(_QWORD *)(*(_QWORD *)(a1 + 640) + v27) = v19;
    *(_DWORD *)(v9 + 64) = 0;
    *(_QWORD *)(v9 + 40) = 0;
  }
  sub_33E4EC0(a1, v9, (__int64)a7, a8);
  v32 = HIDWORD(v59);
  if ( HIDWORD(v59) == v60 )
    goto LABEL_41;
  v55 = 0x1000000000LL;
  v34 = (_QWORD *)v58;
  v54 = v56;
  if ( !v61 )
    v32 = (unsigned int)v59;
  v35 = (_QWORD *)(v58 + 8 * v32);
  if ( (_QWORD *)v58 == v35 )
    goto LABEL_51;
  while ( 1 )
  {
    v36 = *v34;
    v37 = v34;
    if ( *v34 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v35 == ++v34 )
      goto LABEL_51;
  }
  if ( v35 == v34 )
  {
LABEL_51:
    v38 = &v54;
  }
  else
  {
    v38 = &v54;
    if ( !*(_QWORD *)(v36 + 56) )
      goto LABEL_64;
    while ( 1 )
    {
      v39 = v37 + 1;
      if ( v37 + 1 == v35 )
        break;
      v36 = *v39;
      for ( ++v37; *v39 >= 0xFFFFFFFFFFFFFFFELL; v37 = v39 )
      {
        if ( v35 == ++v39 )
          goto LABEL_52;
        v36 = *v39;
      }
      if ( v35 == v37 )
        break;
      if ( !*(_QWORD *)(v36 + 56) )
      {
LABEL_64:
        v40 = (unsigned int)v55;
        v41 = (unsigned int)v55 + 1LL;
        if ( v41 > HIDWORD(v55) )
        {
          v51 = v38;
          sub_C8D5F0((__int64)v38, v56, v41, 8u, (__int64)v38, v31);
          v40 = (unsigned int)v55;
          v38 = v51;
        }
        *(_QWORD *)&v54[8 * v40] = v36;
        LODWORD(v55) = v55 + 1;
      }
    }
  }
LABEL_52:
  sub_33EBD60(a1, (__int64)v38);
  if ( v54 == v56 )
  {
LABEL_41:
    v33 = v53;
    if ( !v53 )
      goto LABEL_43;
    goto LABEL_42;
  }
  _libc_free((unsigned __int64)v54);
  v33 = v53;
  if ( v53 )
LABEL_42:
    sub_C657C0((__int64 *)(a1 + 520), (__int64 *)v9, v33, (__int64)off_4A367D0);
LABEL_43:
  if ( !v61 )
  {
    v15 = v58;
    goto LABEL_45;
  }
  return v9;
}

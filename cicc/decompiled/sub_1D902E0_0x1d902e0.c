// Function: sub_1D902E0
// Address: 0x1d902e0
//
__int64 __fastcall sub_1D902E0(
        __int64 a1,
        unsigned __int8 a2,
        unsigned int a3,
        unsigned __int8 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v12; // r12
  unsigned int v14; // r9d
  __int64 v15; // r15
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdx
  int v19; // edx
  _QWORD *v20; // r13
  _QWORD *v21; // rax
  __int64 v22; // r12
  double v23; // xmm4_8
  double v24; // xmm5_8
  __int64 v25; // rbx
  __int64 *v26; // r14
  _QWORD *v28; // rax
  int v29; // r8d
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 i; // r13
  unsigned __int8 v34; // r15
  char v35; // al
  __int64 v36; // rax
  unsigned __int64 v37; // rdi
  __int64 *v38; // rbx
  __int64 *v39; // r13
  __int64 *v40; // rax
  __int64 v41; // r15
  __int64 v42; // r15
  _QWORD *v43; // rax
  __int64 v44; // r13
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 *v47; // rdx
  __int64 *v48; // rdi
  __int64 *v49; // rsi
  __int64 *v50; // rdx
  __int64 v51; // [rsp+20h] [rbp-210h]
  _QWORD *v52; // [rsp+28h] [rbp-208h]
  __int64 v53; // [rsp+28h] [rbp-208h]
  __int64 v54; // [rsp+28h] [rbp-208h]
  __int64 v55; // [rsp+30h] [rbp-200h]
  unsigned __int8 v56; // [rsp+30h] [rbp-200h]
  unsigned __int8 v58; // [rsp+3Fh] [rbp-1F1h]
  char v59; // [rsp+3Fh] [rbp-1F1h]
  __int64 v60; // [rsp+40h] [rbp-1F0h] BYREF
  __int64 *v61; // [rsp+48h] [rbp-1E8h]
  __int64 *v62; // [rsp+50h] [rbp-1E0h]
  __int64 v63; // [rsp+58h] [rbp-1D8h]
  int v64; // [rsp+60h] [rbp-1D0h]
  _BYTE v65[136]; // [rsp+68h] [rbp-1C8h] BYREF
  __int64 *v66; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v67; // [rsp+F8h] [rbp-138h]
  _BYTE v68[304]; // [rsp+100h] [rbp-130h] BYREF

  v12 = *(_QWORD *)(a1 + 80);
  v66 = (__int64 *)v68;
  v67 = 0x2000000000LL;
  if ( v12 == a1 + 72 )
    return 0;
  v14 = 0;
  do
  {
    if ( !v12 )
      BUG();
    v15 = *(_QWORD *)(v12 + 24);
    v16 = v12 + 16;
    if ( v15 != v12 + 16 )
    {
      v55 = v12;
      while ( 1 )
      {
        while ( 1 )
        {
          v17 = v15;
          v15 = *(_QWORD *)(v15 + 8);
          if ( *(_BYTE *)(v17 - 8) != 78 )
            goto LABEL_6;
          v18 = *(_QWORD *)(v17 - 48);
          if ( *(_BYTE *)(v18 + 16) || (*(_BYTE *)(v18 + 33) & 0x20) == 0 )
            goto LABEL_6;
          v19 = *(_DWORD *)(v18 + 36);
          v20 = (_QWORD *)(v17 - 24);
          if ( v19 == 105 )
          {
            v14 = 1;
            if ( a4 )
            {
              v30 = sub_1649C60(*(_QWORD *)(v17 - 24LL * (*(_DWORD *)(v17 - 4) & 0xFFFFFFF) - 24));
              v31 = (unsigned int)v67;
              if ( (unsigned int)v67 >= HIDWORD(v67) )
              {
                v54 = v30;
                sub_16CD150((__int64)&v66, v68, 0, 8, v29, v30);
                v31 = (unsigned int)v67;
                v30 = v54;
              }
              v66[v31] = v30;
              v14 = a4;
              LODWORD(v67) = v67 + 1;
            }
            goto LABEL_6;
          }
          if ( v19 != 106 )
            break;
          v14 = a3;
          if ( !(_BYTE)a3 )
          {
            v51 = *(_QWORD *)(v17 - 24LL * (*(_DWORD *)(v17 - 4) & 0xFFFFFFF) - 24);
            v53 = v20[3 * (2LL - (*(_DWORD *)(v17 - 4) & 0xFFFFFFF))];
            v28 = sub_1648A60(64, 2u);
            v22 = (__int64)v28;
            if ( v28 )
              sub_15F9660((__int64)v28, v51, v53, (__int64)v20);
            goto LABEL_17;
          }
LABEL_6:
          if ( v16 == v15 )
            goto LABEL_18;
        }
        if ( v19 != 104 )
          goto LABEL_6;
        v14 = a2;
        if ( a2 )
          goto LABEL_6;
        v52 = (_QWORD *)v20[3 * (1LL - (*(_DWORD *)(v17 - 4) & 0xFFFFFFF))];
        v21 = sub_1648A60(64, 1u);
        v22 = (__int64)v21;
        if ( v21 )
          sub_15F9100((__int64)v21, v52, byte_3F871B3, (__int64)v20);
        sub_164B7C0(v22, (__int64)v20);
LABEL_17:
        sub_164D160((__int64)v20, v22, a5, a6, a7, a8, v23, v24, a11, a12);
        sub_15F20C0(v20);
        v14 = 1;
        if ( v16 == v15 )
        {
LABEL_18:
          v12 = v55;
          break;
        }
      }
    }
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( a1 + 72 != v12 );
  v25 = (unsigned int)v67;
  v26 = v66;
  if ( !(_DWORD)v67 )
    goto LABEL_21;
  v32 = *(_QWORD *)(a1 + 80);
  if ( !v32 )
    BUG();
  for ( i = *(_QWORD *)(v32 + 24); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 8) != 53 )
      break;
  }
  v34 = v14;
  v60 = 0;
  v61 = (__int64 *)v65;
  v62 = (__int64 *)v65;
  v63 = 16;
  v64 = 0;
  while ( 1 )
  {
    v35 = *(_BYTE *)(i - 8);
    if ( v35 == 53 )
      goto LABEL_66;
    if ( (unsigned __int8)(v35 - 54) > 2u )
      break;
    if ( v35 != 55 )
      goto LABEL_66;
    v46 = sub_1649C60(*(_QWORD *)(i - 48));
    if ( *(_BYTE *)(v46 + 16) != 53 )
      goto LABEL_66;
    v47 = v61;
    if ( v62 == v61 )
    {
      v48 = &v61[HIDWORD(v63)];
      if ( v61 == v48 )
        goto LABEL_84;
      v49 = 0;
      do
      {
        if ( v46 == *v47 )
          goto LABEL_66;
        if ( *v47 == -2 )
          v49 = v47;
        ++v47;
      }
      while ( v48 != v47 );
      if ( !v49 )
      {
LABEL_84:
        if ( HIDWORD(v63) >= (unsigned int)v63 )
          goto LABEL_69;
        ++HIDWORD(v63);
        *v48 = v46;
        ++v60;
      }
      else
      {
        *v49 = v46;
        --v64;
        ++v60;
      }
LABEL_66:
      i = *(_QWORD *)(i + 8);
      if ( !i )
        goto LABEL_90;
    }
    else
    {
LABEL_69:
      sub_16CCBA0((__int64)&v60, v46);
      i = *(_QWORD *)(i + 8);
      if ( !i )
LABEL_90:
        BUG();
    }
  }
  if ( v35 == 78 )
  {
    v36 = *(_QWORD *)(i - 48);
    if ( !*(_BYTE *)(v36 + 16) && *(_DWORD *)(v36 + 36) == 105 )
      goto LABEL_66;
  }
  v59 = 0;
  v37 = (unsigned __int64)v62;
  v38 = &v26[v25];
  v56 = v34;
  do
  {
    while ( 1 )
    {
      v40 = v61;
      v41 = *v26;
      if ( (__int64 *)v37 != v61 )
        break;
      v39 = (__int64 *)(v37 + 8LL * HIDWORD(v63));
      if ( (__int64 *)v37 == v39 )
      {
        v50 = (__int64 *)v37;
      }
      else
      {
        do
        {
          if ( v41 == *v40 )
            break;
          ++v40;
        }
        while ( v39 != v40 );
        v50 = (__int64 *)(v37 + 8LL * HIDWORD(v63));
      }
LABEL_56:
      while ( v50 != v40 )
      {
        if ( (unsigned __int64)*v40 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_46;
        ++v40;
      }
      if ( v39 == v40 )
        goto LABEL_58;
LABEL_47:
      if ( v38 == ++v26 )
        goto LABEL_61;
    }
    v39 = (__int64 *)(v37 + 8LL * (unsigned int)v63);
    v40 = sub_16CC9F0((__int64)&v60, *v26);
    if ( v41 == *v40 )
    {
      v37 = (unsigned __int64)v62;
      if ( v62 == v61 )
        v50 = &v62[HIDWORD(v63)];
      else
        v50 = &v62[(unsigned int)v63];
      goto LABEL_56;
    }
    v37 = (unsigned __int64)v62;
    if ( v62 == v61 )
    {
      v40 = &v62[HIDWORD(v63)];
      v50 = v40;
      goto LABEL_56;
    }
    v40 = &v62[(unsigned int)v63];
LABEL_46:
    if ( v39 != v40 )
      goto LABEL_47;
LABEL_58:
    v42 = sub_1599A20(*(__int64 ***)(*v26 + 56));
    v43 = sub_1648A60(64, 2u);
    v44 = (__int64)v43;
    if ( v43 )
      sub_15F9650((__int64)v43, v42, *v26, 0, 0);
    v45 = *v26++;
    sub_15F2180(v44, v45);
    v59 = 1;
    v37 = (unsigned __int64)v62;
  }
  while ( v38 != v26 );
LABEL_61:
  v14 = v56;
  if ( (__int64 *)v37 != v61 )
  {
    _libc_free(v37);
    v14 = v56;
  }
  v26 = v66;
  LOBYTE(v14) = v59 | v14;
LABEL_21:
  if ( v26 != (__int64 *)v68 )
  {
    v58 = v14;
    _libc_free((unsigned __int64)v26);
    return v58;
  }
  return v14;
}

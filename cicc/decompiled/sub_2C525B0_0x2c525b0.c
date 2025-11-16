// Function: sub_2C525B0
// Address: 0x2c525b0
//
__int64 __fastcall sub_2C525B0(int **a1, __int64 *a2, int a3, int *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v9; // r15
  _BYTE *v11; // rsi
  int v13; // edi
  char v14; // al
  __int64 v15; // rsi
  _BYTE *v16; // rdi
  __int64 v17; // r14
  _DWORD *v18; // r11
  __int64 v19; // rdi
  _DWORD *v20; // r10
  int v21; // esi
  __int64 v22; // r13
  __int64 v23; // rdi
  _DWORD *v24; // rax
  _DWORD *v25; // rdi
  int *v26; // rsi
  int v27; // eax
  int v28; // eax
  __int64 **v29; // r11
  __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // r13
  __int64 *v36; // rsi
  int v37; // edi
  __int64 *v38; // rcx
  unsigned __int64 v39; // rax
  __int64 v40; // r13
  int v41; // edx
  int v42; // r14d
  int *v43; // rax
  unsigned __int64 v44; // rdx
  int v45; // [rsp+0h] [rbp-120h]
  char v46; // [rsp+8h] [rbp-118h]
  __int64 v47; // [rsp+8h] [rbp-118h]
  unsigned int v48; // [rsp+10h] [rbp-110h]
  unsigned int v49; // [rsp+10h] [rbp-110h]
  __int64 v50; // [rsp+10h] [rbp-110h]
  unsigned int v51; // [rsp+10h] [rbp-110h]
  __int64 v52; // [rsp+18h] [rbp-108h]
  __int64 v53; // [rsp+18h] [rbp-108h]
  __int64 **v54; // [rsp+18h] [rbp-108h]
  __int64 v55; // [rsp+18h] [rbp-108h]
  int *v56; // [rsp+20h] [rbp-100h]
  int *v57; // [rsp+20h] [rbp-100h]
  __int64 v58; // [rsp+20h] [rbp-100h]
  int *v59; // [rsp+20h] [rbp-100h]
  int v60; // [rsp+28h] [rbp-F8h]
  int v61; // [rsp+28h] [rbp-F8h]
  int v62; // [rsp+28h] [rbp-F8h]
  _QWORD v63[2]; // [rsp+30h] [rbp-F0h] BYREF
  _BYTE *v64; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v65; // [rsp+48h] [rbp-D8h]
  _BYTE v66[64]; // [rsp+50h] [rbp-D0h] BYREF
  __int64 *v67; // [rsp+90h] [rbp-90h] BYREF
  unsigned __int64 v68; // [rsp+98h] [rbp-88h]
  __int64 v69; // [rsp+A0h] [rbp-80h] BYREF
  int v70; // [rsp+A8h] [rbp-78h]
  char v71; // [rsp+ACh] [rbp-74h]
  char v72; // [rsp+B0h] [rbp-70h] BYREF

  v6 = *a2;
  v7 = *(_QWORD *)(*a2 + 16);
  if ( !v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 8) )
    return 0;
  if ( *(_BYTE *)v6 != 92 )
    return 0;
  v9 = *(_QWORD *)(v6 - 64);
  if ( !v9 )
    return 0;
  v11 = *(_BYTE **)(v6 - 32);
  v13 = (unsigned __int8)*v11;
  if ( (_BYTE)v13 == 12 || v13 == 13 )
  {
    v17 = v6;
  }
  else
  {
    v48 = a6;
    v52 = a5;
    v56 = a4;
    v60 = a3;
    if ( (unsigned __int8)(*v11 - 9) > 2u )
      return 0;
    v67 = 0;
    v68 = (unsigned __int64)&v72;
    v64 = v66;
    v63[0] = &v67;
    v65 = 0x800000000LL;
    v69 = 8;
    v70 = 0;
    v71 = 1;
    v63[1] = &v64;
    v14 = sub_AA8FD0(v63, (__int64)v11);
    a3 = v60;
    a4 = v56;
    v46 = v14;
    a5 = v52;
    a6 = v48;
    if ( v14 )
    {
      do
      {
        v16 = v64;
        if ( !(_DWORD)v65 )
        {
          a3 = v60;
          a4 = v56;
          a5 = v52;
          a6 = v48;
          goto LABEL_14;
        }
        v15 = *(_QWORD *)&v64[8 * (unsigned int)v65 - 8];
        LODWORD(v65) = v65 - 1;
      }
      while ( (unsigned __int8)sub_AA8FD0(v63, v15) );
      a3 = v60;
      a4 = v56;
      a5 = v52;
      a6 = v48;
    }
    v46 = 0;
    v16 = v64;
LABEL_14:
    if ( v16 != v66 )
    {
      v49 = a6;
      v53 = a5;
      v57 = a4;
      v61 = a3;
      _libc_free((unsigned __int64)v16);
      a6 = v49;
      a5 = v53;
      a4 = v57;
      a3 = v61;
    }
    if ( !v71 )
    {
      v51 = a6;
      v55 = a5;
      v59 = a4;
      v62 = a3;
      _libc_free(v68);
      a6 = v51;
      a5 = v55;
      a4 = v59;
      a3 = v62;
    }
    if ( !v46 )
      return 0;
    v17 = *a2;
  }
  if ( *(_QWORD *)(v9 + 8) != *(_QWORD *)(v17 + 8) )
    return 0;
  v18 = *(_DWORD **)(v6 + 72);
  v19 = 4LL * *(unsigned int *)(v6 + 80);
  v20 = &v18[(unsigned __int64)v19 / 4];
  v21 = **a1;
  v22 = v19 >> 2;
  v23 = v19 >> 4;
  if ( v23 )
  {
    v24 = v18;
    v25 = &v18[4 * v23];
    while ( *v24 < v21 )
    {
      if ( v24[1] >= v21 )
      {
        ++v24;
        goto LABEL_28;
      }
      if ( v24[2] >= v21 )
      {
        v24 += 2;
        goto LABEL_28;
      }
      if ( v24[3] >= v21 )
      {
        v24 += 3;
        goto LABEL_28;
      }
      v24 += 4;
      if ( v25 == v24 )
      {
        v22 = v20 - v24;
        goto LABEL_55;
      }
    }
    goto LABEL_28;
  }
  v24 = v18;
LABEL_55:
  if ( v22 == 2 )
    goto LABEL_65;
  if ( v22 == 3 )
  {
    if ( *v24 >= v21 )
      goto LABEL_28;
    ++v24;
LABEL_65:
    if ( *v24 >= v21 )
      goto LABEL_28;
    ++v24;
    goto LABEL_58;
  }
  if ( v22 != 1 )
    goto LABEL_29;
LABEL_58:
  if ( *v24 < v21 )
    goto LABEL_29;
LABEL_28:
  if ( v20 != v24 )
    return 0;
LABEL_29:
  v26 = &a4[a5];
  if ( v26 != a4 )
  {
    do
    {
      v27 = *a4;
      if ( *a4 >= a3 && v27 < a3 + **a1 )
      {
        v28 = v18[v27 - a3];
        if ( v28 >= 0 )
          v28 += a3;
        *a4 = v28;
      }
      ++a4;
    }
    while ( v26 != a4 );
    v17 = *a2;
  }
  v29 = (__int64 **)*((_QWORD *)a1[2] + 19);
  v30 = 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(v17 + 7) & 0x40) != 0 )
  {
    v31 = *(_QWORD *)(v17 - 8);
    v32 = v31 + v30;
  }
  else
  {
    v31 = v17 - v30;
    v32 = v17;
  }
  v33 = v32 - v31;
  v68 = 0x400000000LL;
  v34 = v33 >> 5;
  v67 = &v69;
  v35 = v33 >> 5;
  if ( (unsigned __int64)v33 > 0x80 )
  {
    v45 = a6;
    v47 = v33;
    v50 = v31;
    v54 = v29;
    v58 = v33 >> 5;
    sub_C8D5F0((__int64)&v67, &v69, v34, 8u, v31, a6);
    v38 = v67;
    v37 = v68;
    LODWORD(v34) = v58;
    v29 = v54;
    v31 = v50;
    v36 = &v67[(unsigned int)v68];
    v33 = v47;
    LODWORD(a6) = v45;
  }
  else
  {
    v36 = &v69;
    v37 = 0;
    v38 = &v69;
  }
  if ( v33 > 0 )
  {
    v39 = 0;
    do
    {
      v36[v39 / 8] = *(_QWORD *)(v31 + 4 * v39);
      v39 += 8LL;
      --v35;
    }
    while ( v35 );
    v38 = v67;
    v37 = v68;
  }
  LODWORD(v68) = v37 + v34;
  v40 = sub_DFCEF0(v29, (unsigned __int8 *)v17, (unsigned __int8 **)v38, (unsigned int)(v37 + v34), a6);
  v42 = v41;
  if ( v67 != &v69 )
    _libc_free((unsigned __int64)v67);
  v43 = a1[1];
  if ( v42 == 1 )
    v43[2] = 1;
  v44 = *(_QWORD *)v43 + v40;
  if ( __OFADD__(*(_QWORD *)v43, v40) )
  {
    v44 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v40 <= 0 )
      v44 = 0x8000000000000000LL;
  }
  *(_QWORD *)v43 = v44;
  *a2 = v9;
  return 1;
}

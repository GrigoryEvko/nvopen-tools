// Function: sub_1427500
// Address: 0x1427500
//
void __fastcall sub_1427500(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rsi
  int v6; // r14d
  __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rbx
  __int64 v11; // r8
  int v12; // eax
  bool v13; // zf
  unsigned __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r15
  __int64 v19; // rcx
  __int64 v20; // r13
  __int64 *v21; // rbx
  __int64 *v22; // r14
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rbx
  unsigned __int64 v35; // rdi
  __int64 v36; // r13
  _QWORD *v37; // rbx
  _QWORD *v38; // rax
  __int64 v39; // r15
  __int64 v40; // rax
  __int64 *v41; // rax
  __int64 v42; // rax
  __int64 *v43; // rsi
  __int64 *v44; // rcx
  _QWORD *v45; // rdx
  __int64 v46; // [rsp+8h] [rbp-258h]
  __int64 v47; // [rsp+10h] [rbp-250h]
  __int64 i; // [rsp+18h] [rbp-248h]
  char v49; // [rsp+20h] [rbp-240h]
  __int64 v50; // [rsp+20h] [rbp-240h]
  __int64 v51; // [rsp+28h] [rbp-238h]
  __int64 v52; // [rsp+28h] [rbp-238h]
  _QWORD v53[4]; // [rsp+30h] [rbp-230h] BYREF
  __int64 v54; // [rsp+50h] [rbp-210h] BYREF
  _BYTE *v55; // [rsp+58h] [rbp-208h]
  _BYTE *v56; // [rsp+60h] [rbp-200h]
  __int64 v57; // [rsp+68h] [rbp-1F8h]
  int v58; // [rsp+70h] [rbp-1F0h]
  _BYTE v59[136]; // [rsp+78h] [rbp-1E8h] BYREF
  __int64 v60; // [rsp+100h] [rbp-160h] BYREF
  __int64 *v61; // [rsp+108h] [rbp-158h]
  __int64 *v62; // [rsp+110h] [rbp-150h]
  __int64 v63; // [rsp+118h] [rbp-148h]
  int v64; // [rsp+120h] [rbp-140h]
  _BYTE v65[312]; // [rsp+128h] [rbp-138h] BYREF

  v2 = *(_QWORD *)(a1 + 16);
  v3 = *(_QWORD *)(v2 + 80);
  if ( v3 )
    v3 -= 24;
  v4 = sub_15E0530(v2);
  v5 = 1;
  v6 = *(_DWORD *)(a1 + 336);
  v7 = v4;
  *(_DWORD *)(a1 + 336) = v6 + 1;
  v10 = sub_1648A60(120, 1);
  if ( v10 )
  {
    v5 = sub_1643270(v7);
    sub_1648CB0(v10, v5, 22);
    v12 = *(_DWORD *)(v10 + 20);
    *(_QWORD *)(v10 + 32) = 0;
    *(_QWORD *)(v10 + 40) = 0;
    *(_QWORD *)(v10 + 48) = 0;
    v13 = *(_QWORD *)(v10 - 24) == 0;
    *(_QWORD *)(v10 + 56) = 0;
    *(_DWORD *)(v10 + 20) = v12 & 0xF0000000 | 1;
    *(_QWORD *)(v10 + 64) = v3;
    *(_QWORD *)(v10 + 24) = sub_141FFD0;
    *(_QWORD *)(v10 + 72) = 0;
    *(_WORD *)(v10 + 80) = 257;
    if ( !v13 )
    {
      v8 = *(_QWORD *)(v10 - 16);
      v14 = *(_QWORD *)(v10 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v14 = v8;
      if ( v8 )
      {
        v9 = *(_QWORD *)(v8 + 16) & 3LL;
        *(_QWORD *)(v8 + 16) = v9 | v14;
      }
    }
    *(_QWORD *)(v10 - 24) = 0;
    *(_DWORD *)(v10 + 84) = v6;
    *(_DWORD *)(v10 + 88) = -1;
    *(_QWORD *)(v10 + 96) = 4;
    *(_QWORD *)(v10 + 104) = 0;
    *(_QWORD *)(v10 + 112) = 0;
  }
  v15 = *(_QWORD *)(a1 + 120);
  *(_QWORD *)(a1 + 120) = v10;
  if ( v15 )
    sub_164BEC0(v15, v5, v8, v9, v11);
  v60 = 0;
  v61 = (__int64 *)v65;
  v62 = (__int64 *)v65;
  v16 = *(_QWORD *)(a1 + 16);
  v63 = 32;
  v17 = *(_QWORD *)(v16 + 80);
  v64 = 0;
  v46 = v16 + 72;
  for ( i = v17; v46 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v18 = *(_QWORD *)(i + 24);
    v19 = i - 24;
    v20 = i + 16;
    v21 = 0;
    v51 = i - 24;
    v22 = 0;
    v49 = 0;
    if ( i + 16 == v18 )
      continue;
    do
    {
      while ( 1 )
      {
        v23 = v18 - 24;
        if ( !v18 )
          v23 = 0;
        v24 = sub_1427030(a1, v23, v8, v19);
        v8 = v24;
        if ( v24 )
        {
          if ( !v22 )
          {
            v47 = v24;
            v40 = sub_1425DF0(a1, v51);
            v8 = v47;
            v22 = (__int64 *)v40;
          }
          v25 = *v22;
          v26 = *(_QWORD *)(v8 + 32);
          *(_QWORD *)(v8 + 40) = v22;
          v25 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v8 + 32) = v25 | v26 & 7;
          *(_QWORD *)(v25 + 8) = v8 + 32;
          *v22 = *v22 & 7 | (v8 + 32);
          if ( *(_BYTE *)(v8 + 16) == 22 )
            break;
        }
        v18 = *(_QWORD *)(v18 + 8);
        if ( v20 == v18 )
          goto LABEL_24;
      }
      if ( !v21 )
      {
        v50 = v8;
        v42 = sub_1426290(a1, v51);
        v8 = v50;
        v21 = (__int64 *)v42;
      }
      v27 = *v21;
      v28 = *(_QWORD *)(v8 + 48);
      *(_QWORD *)(v8 + 56) = v21;
      v49 = 1;
      v27 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v8 + 48) = v27 | v28 & 7;
      *(_QWORD *)(v27 + 8) = v8 + 48;
      *v21 = *v21 & 7 | (v8 + 48);
      v18 = *(_QWORD *)(v18 + 8);
    }
    while ( v20 != v18 );
LABEL_24:
    if ( !v49 )
      continue;
    v41 = v61;
    if ( v62 != v61 )
      goto LABEL_52;
    v8 = HIDWORD(v63);
    v43 = &v61[HIDWORD(v63)];
    if ( v61 != v43 )
    {
      v44 = 0;
      while ( 1 )
      {
        v8 = *v41;
        if ( v51 == *v41 )
          break;
        if ( v8 == -2 )
          v44 = v41;
        if ( v43 == ++v41 )
        {
          if ( !v44 )
            goto LABEL_65;
          *v44 = v51;
          --v64;
          ++v60;
          goto LABEL_25;
        }
      }
      continue;
    }
LABEL_65:
    if ( HIDWORD(v63) < (unsigned int)v63 )
    {
      ++HIDWORD(v63);
      *v43 = v51;
      ++v60;
    }
    else
    {
LABEL_52:
      sub_16CCBA0(&v60, v51);
    }
LABEL_25:
    ;
  }
  sub_1426F70(a1, (__int64)&v60);
  v29 = *(_QWORD *)(a1 + 120);
  v54 = 0;
  v55 = v59;
  v56 = v59;
  v30 = *(_QWORD *)(a1 + 8);
  v57 = 16;
  v58 = 0;
  sub_1421630(a1, *(__int64 **)(v30 + 56), v29, (__int64)&v54, 0, 0);
  v31 = sub_1423AE0((__int64 *)a1);
  v32 = *(_QWORD *)(a1 + 8);
  v33 = *(_QWORD *)a1;
  v53[0] = a1;
  v53[2] = v33;
  v53[3] = v32;
  v53[1] = v31;
  sub_1423BA0((__int64 *)a1);
  sub_14246A0(v53);
  v34 = *(_QWORD *)(a1 + 16);
  v35 = (unsigned __int64)v56;
  v36 = *(_QWORD *)(v34 + 80);
  v52 = v34 + 72;
  if ( v34 + 72 != v36 )
  {
    while ( 1 )
    {
      v39 = v36 - 24;
      if ( !v36 )
        v39 = 0;
      v38 = v55;
      if ( (_BYTE *)v35 == v55 )
        break;
      v37 = (_QWORD *)(v35 + 8LL * (unsigned int)v57);
      v38 = (_QWORD *)sub_16CC9F0(&v54, v39);
      if ( v39 == *v38 )
      {
        v35 = (unsigned __int64)v56;
        if ( v56 == v55 )
          v45 = &v56[8 * HIDWORD(v57)];
        else
          v45 = &v56[8 * (unsigned int)v57];
        goto LABEL_41;
      }
      v35 = (unsigned __int64)v56;
      if ( v56 == v55 )
      {
        v38 = &v56[8 * HIDWORD(v57)];
        v45 = v38;
        goto LABEL_41;
      }
      v38 = &v56[8 * (unsigned int)v57];
LABEL_31:
      if ( v38 == v37 )
      {
LABEL_45:
        sub_1421A90(a1, v39);
        v35 = (unsigned __int64)v56;
        v36 = *(_QWORD *)(v36 + 8);
        if ( v52 == v36 )
          goto LABEL_46;
      }
      else
      {
LABEL_32:
        v36 = *(_QWORD *)(v36 + 8);
        if ( v52 == v36 )
          goto LABEL_46;
      }
    }
    v37 = (_QWORD *)(v35 + 8LL * HIDWORD(v57));
    if ( (_QWORD *)v35 == v37 )
    {
      v45 = (_QWORD *)v35;
    }
    else
    {
      do
      {
        if ( v39 == *v38 )
          break;
        ++v38;
      }
      while ( v37 != v38 );
      v45 = (_QWORD *)(v35 + 8LL * HIDWORD(v57));
    }
LABEL_41:
    if ( v38 != v45 )
    {
      while ( *v38 >= 0xFFFFFFFFFFFFFFFELL )
      {
        if ( v45 == ++v38 )
        {
          if ( v38 != v37 )
            goto LABEL_32;
          goto LABEL_45;
        }
      }
    }
    goto LABEL_31;
  }
LABEL_46:
  if ( (_BYTE *)v35 != v55 )
    _libc_free(v35);
  if ( v62 != v61 )
    _libc_free((unsigned __int64)v62);
}

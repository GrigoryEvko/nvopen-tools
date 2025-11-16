// Function: sub_150A160
// Address: 0x150a160
//
_QWORD *__fastcall sub_150A160(
        __int64 a1,
        int a2,
        char *a3,
        size_t a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned int v10; // r14d
  __int64 v11; // r15
  __int64 v12; // rdx
  __int64 v13; // r14
  bool v14; // zf
  _QWORD *v15; // rax
  _QWORD *v16; // rbx
  _QWORD *v17; // r15
  _QWORD *v18; // r12
  unsigned __int64 v19; // r13
  __int64 v20; // r8
  unsigned int v21; // esi
  unsigned __int64 v22; // r13
  __int64 v23; // rdi
  unsigned int v24; // ecx
  int *v25; // rax
  int v26; // edx
  _QWORD *result; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rdi
  _BYTE *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r14
  _BYTE *v36; // rdi
  _BYTE *v37; // rax
  _QWORD *v38; // rsi
  __int64 v39; // rdx
  size_t v40; // rdx
  __int64 v41; // rax
  int v42; // r14d
  int *v43; // r11
  int v44; // edx
  int v45; // ecx
  int v46; // ecx
  __int64 v47; // r8
  unsigned int v48; // esi
  int v49; // edi
  int v50; // r10d
  int *v51; // r9
  int v52; // ecx
  int v53; // ecx
  __int64 v54; // rdi
  int *v55; // r8
  unsigned int v56; // ebx
  int v57; // r9d
  int v58; // esi
  __int64 v59; // [rsp+8h] [rbp-158h]
  size_t v60; // [rsp+10h] [rbp-150h]
  __int64 v61; // [rsp+18h] [rbp-148h]
  char v62; // [rsp+27h] [rbp-139h]
  __int64 v64; // [rsp+30h] [rbp-130h]
  __int64 v65; // [rsp+30h] [rbp-130h]
  __int64 v67; // [rsp+48h] [rbp-118h] BYREF
  _QWORD v68[2]; // [rsp+50h] [rbp-110h] BYREF
  _QWORD v69[2]; // [rsp+60h] [rbp-100h] BYREF
  _QWORD *v70; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v71; // [rsp+78h] [rbp-E8h]
  _QWORD v72[2]; // [rsp+80h] [rbp-E0h] BYREF
  __m128i v73; // [rsp+90h] [rbp-D0h] BYREF
  _QWORD *v74; // [rsp+A0h] [rbp-C0h]
  _QWORD *v75; // [rsp+A8h] [rbp-B8h]
  __int64 v76; // [rsp+B0h] [rbp-B0h]

  v59 = (__int64)a3;
  v60 = a4;
  v10 = a5 - 7;
  sub_15E4CF0(v68, a3, a4, a5, a7, a8);
  v11 = v68[1];
  v64 = v68[0];
  sub_16C1840(&v73);
  sub_16C1A90(&v73, v64, v11);
  sub_16C1AA0(&v73, &v70);
  v61 = (__int64)v70;
  v65 = (__int64)v70;
  if ( v10 <= 1 )
  {
    sub_16C1840(&v73);
    sub_16C1A90(&v73, a3, a4);
    sub_16C1AA0(&v73, &v70);
    v65 = (__int64)v70;
  }
  if ( byte_4F9DBC0 )
  {
    v28 = sub_16BA580(&v73, &v70, v12);
    v29 = *(_QWORD *)(v28 + 24);
    v30 = v28;
    if ( (unsigned __int64)(*(_QWORD *)(v28 + 16) - v29) <= 4 )
    {
      v30 = sub_16E7EE0(v28, "GUID ", 5);
    }
    else
    {
      *(_DWORD *)v29 = 1145656647;
      *(_BYTE *)(v29 + 4) = 32;
      *(_QWORD *)(v28 + 24) += 5LL;
    }
    v31 = sub_16E7A90(v30, v61);
    v32 = *(_BYTE **)(v31 + 24);
    if ( *(_BYTE **)(v31 + 16) == v32 )
    {
      v31 = sub_16E7EE0(v31, "(", 1);
    }
    else
    {
      *v32 = 40;
      ++*(_QWORD *)(v31 + 24);
    }
    v33 = sub_16E7A90(v31, v65);
    v34 = *(_QWORD *)(v33 + 24);
    v35 = v33;
    if ( (unsigned __int64)(*(_QWORD *)(v33 + 16) - v34) <= 4 )
    {
      v41 = sub_16E7EE0(v33, ") is ", 5);
      v36 = *(_BYTE **)(v41 + 24);
      v35 = v41;
    }
    else
    {
      *(_DWORD *)v34 = 1936269353;
      *(_BYTE *)(v34 + 4) = 32;
      v36 = (_BYTE *)(*(_QWORD *)(v33 + 24) + 5LL);
      *(_QWORD *)(v33 + 24) = v36;
    }
    v37 = *(_BYTE **)(v35 + 16);
    if ( v37 - v36 < a4 )
    {
      v35 = sub_16E7EE0(v35, a3, a4);
      v37 = *(_BYTE **)(v35 + 16);
      v36 = *(_BYTE **)(v35 + 24);
    }
    else if ( a4 )
    {
      memcpy(v36, a3, a4);
      v37 = *(_BYTE **)(v35 + 16);
      v36 = (_BYTE *)(a4 + *(_QWORD *)(v35 + 24));
      *(_QWORD *)(v35 + 24) = v36;
    }
    if ( v37 == v36 )
    {
      sub_16E7EE0(v35, "\n", 1);
    }
    else
    {
      *v36 = 10;
      ++*(_QWORD *)(v35 + 24);
    }
  }
  v62 = 0;
  v13 = *(_QWORD *)(a1 + 424);
  if ( !*(_BYTE *)(a1 + 384) )
  {
    if ( a3 )
    {
      v70 = v72;
      sub_14E9CA0((__int64 *)&v70, a3, (__int64)&a3[a4]);
      v38 = v70;
      v39 = v71;
    }
    else
    {
      v38 = v72;
      LOBYTE(v72[0]) = 0;
      v39 = 0;
      v70 = v72;
      v71 = 0;
    }
    v62 = 1;
    v59 = sub_16D3940(v13 + 384, v38, v39);
    v60 = v40;
  }
  v14 = *(_BYTE *)(v13 + 178) == 0;
  v67 = v61;
  if ( v14 )
  {
    v73.m128i_i64[1] = 0;
    v73.m128i_i64[0] = (__int64)byte_3F871B3;
  }
  else
  {
    v73.m128i_i64[0] = 0;
  }
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v15 = sub_142DA40((_QWORD *)v13, (unsigned __int64 *)&v67, &v73);
  v16 = v75;
  v17 = v74;
  v18 = v15;
  v19 = (unsigned __int64)(v15 + 4);
  if ( v75 != v74 )
  {
    do
    {
      if ( *v17 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v17 + 8LL))(*v17);
      ++v17;
    }
    while ( v16 != v17 );
    v17 = v74;
  }
  if ( v17 )
    j_j___libc_free_0(v17, v76 - (_QWORD)v17);
  v18[5] = v59;
  v20 = a1 + 448;
  v18[6] = v60;
  v21 = *(_DWORD *)(a1 + 472);
  v22 = (4LL * *(unsigned __int8 *)(v13 + 178)) | v19 & 0xFFFFFFFFFFFFFFFBLL;
  if ( !v21 )
  {
    ++*(_QWORD *)(a1 + 448);
    goto LABEL_53;
  }
  v23 = *(_QWORD *)(a1 + 456);
  v24 = (v21 - 1) & (37 * a2);
  v25 = (int *)(v23 + 24LL * v24);
  v26 = *v25;
  if ( a2 == *v25 )
    goto LABEL_16;
  v42 = 1;
  v43 = 0;
  while ( v26 != -1 )
  {
    if ( !v43 && v26 == -2 )
      v43 = v25;
    v24 = (v21 - 1) & (v42 + v24);
    v25 = (int *)(v23 + 24LL * v24);
    v26 = *v25;
    if ( a2 == *v25 )
      goto LABEL_16;
    ++v42;
  }
  if ( v43 )
    v25 = v43;
  ++*(_QWORD *)(a1 + 448);
  v44 = *(_DWORD *)(a1 + 464) + 1;
  if ( 4 * v44 >= 3 * v21 )
  {
LABEL_53:
    sub_1509F90(v20, 2 * v21);
    v45 = *(_DWORD *)(a1 + 472);
    if ( v45 )
    {
      v46 = v45 - 1;
      v47 = *(_QWORD *)(a1 + 456);
      v44 = *(_DWORD *)(a1 + 464) + 1;
      v48 = v46 & (37 * a2);
      v25 = (int *)(v47 + 24LL * v48);
      v49 = *v25;
      if ( a2 != *v25 )
      {
        v50 = 1;
        v51 = 0;
        while ( v49 != -1 )
        {
          if ( v49 == -2 && !v51 )
            v51 = v25;
          v48 = v46 & (v50 + v48);
          v25 = (int *)(v47 + 24LL * v48);
          v49 = *v25;
          if ( a2 == *v25 )
            goto LABEL_49;
          ++v50;
        }
        if ( v51 )
          v25 = v51;
      }
      goto LABEL_49;
    }
    goto LABEL_81;
  }
  if ( v21 - *(_DWORD *)(a1 + 468) - v44 <= v21 >> 3 )
  {
    sub_1509F90(v20, v21);
    v52 = *(_DWORD *)(a1 + 472);
    if ( v52 )
    {
      v53 = v52 - 1;
      v54 = *(_QWORD *)(a1 + 456);
      v55 = 0;
      v56 = v53 & (37 * a2);
      v57 = 1;
      v44 = *(_DWORD *)(a1 + 464) + 1;
      v25 = (int *)(v54 + 24LL * v56);
      v58 = *v25;
      if ( a2 != *v25 )
      {
        while ( v58 != -1 )
        {
          if ( !v55 && v58 == -2 )
            v55 = v25;
          v56 = v53 & (v57 + v56);
          v25 = (int *)(v54 + 24LL * v56);
          v58 = *v25;
          if ( a2 == *v25 )
            goto LABEL_49;
          ++v57;
        }
        if ( v55 )
          v25 = v55;
      }
      goto LABEL_49;
    }
LABEL_81:
    ++*(_DWORD *)(a1 + 464);
    BUG();
  }
LABEL_49:
  *(_DWORD *)(a1 + 464) = v44;
  if ( *v25 != -1 )
    --*(_DWORD *)(a1 + 468);
  *((_QWORD *)v25 + 1) = 0;
  *((_QWORD *)v25 + 2) = 0;
  *v25 = a2;
LABEL_16:
  *((_QWORD *)v25 + 1) = v22;
  *((_QWORD *)v25 + 2) = v65;
  if ( v62 && v70 != v72 )
    j_j___libc_free_0(v70, v72[0] + 1LL);
  result = v69;
  if ( (_QWORD *)v68[0] != v69 )
    return (_QWORD *)j_j___libc_free_0(v68[0], v69[0] + 1LL);
  return result;
}

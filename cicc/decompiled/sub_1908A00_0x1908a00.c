// Function: sub_1908A00
// Address: 0x1908a00
//
__int64 __fastcall sub_1908A00(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  _QWORD *v11; // r12
  __int64 v12; // rdi
  int *v13; // r13
  __int64 v14; // r15
  __int64 v15; // r8
  char v16; // r13
  unsigned __int64 v17; // rsi
  _QWORD *v18; // rdx
  unsigned __int64 v19; // rcx
  _QWORD *v20; // rax
  char v21; // di
  _BOOL4 v22; // r8d
  _QWORD *v23; // rax
  _QWORD *v24; // rbx
  _QWORD *v25; // r10
  _QWORD *v26; // rax
  unsigned __int64 v27; // r14
  _QWORD *v28; // r15
  char v29; // bl
  _QWORD *v30; // r13
  unsigned __int64 v31; // rcx
  _QWORD *v32; // rdx
  unsigned __int64 v33; // rsi
  _QWORD *v34; // rax
  char v35; // di
  _BOOL4 v36; // r9d
  _QWORD *v37; // rsi
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r13
  __int64 v42; // r12
  __int64 v43; // rbx
  __int64 v44; // rdi
  __int64 v45; // rdi
  int v46; // eax
  __int64 v47; // rdx
  _QWORD *v48; // rax
  _QWORD *i; // rdx
  __int64 v50; // rax
  void *v51; // rdi
  unsigned int v52; // eax
  __int64 v53; // rdx
  double v54; // xmm4_8
  double v55; // xmm5_8
  __int64 result; // rax
  unsigned int v57; // ecx
  _QWORD *v58; // rdi
  unsigned int v59; // eax
  int v60; // eax
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  int v63; // ebx
  __int64 v64; // r12
  _QWORD *v65; // rax
  __int64 v66; // rdx
  _QWORD *j; // rdx
  __int64 v68; // rax
  _QWORD *v69; // rax
  __int64 v71; // [rsp+8h] [rbp-A8h]
  _QWORD *v72; // [rsp+8h] [rbp-A8h]
  _BOOL4 v73; // [rsp+18h] [rbp-98h]
  _BOOL4 v74; // [rsp+18h] [rbp-98h]
  _QWORD *v75; // [rsp+18h] [rbp-98h]
  __int64 v76; // [rsp+18h] [rbp-98h]
  _QWORD *v77; // [rsp+20h] [rbp-90h]
  _QWORD *v78; // [rsp+20h] [rbp-90h]
  unsigned __int64 v79; // [rsp+20h] [rbp-90h]
  unsigned __int8 v80; // [rsp+28h] [rbp-88h]
  __int64 v81; // [rsp+30h] [rbp-80h] BYREF
  __int64 v82; // [rsp+38h] [rbp-78h]
  unsigned __int64 v83; // [rsp+40h] [rbp-70h]
  int v84; // [rsp+58h] [rbp-58h] BYREF
  __int64 v85; // [rsp+60h] [rbp-50h]
  int *v86; // [rsp+68h] [rbp-48h]
  int *v87; // [rsp+70h] [rbp-40h]
  __int64 v88; // [rsp+78h] [rbp-38h]

  v10 = a1;
  v11 = (_QWORD *)(a1 + 168);
  v12 = *(_QWORD *)(a1 + 176);
  v84 = 0;
  v85 = 0;
  v86 = &v84;
  v87 = &v84;
  v88 = 0;
  sub_1903F40(v12);
  v13 = v86;
  *(_QWORD *)(v10 + 176) = 0;
  *(_QWORD *)(v10 + 184) = v11;
  *(_QWORD *)(v10 + 192) = v11;
  v14 = (__int64)v13;
  for ( *(_QWORD *)(v10 + 200) = 0; (int *)v14 != &v84; v14 = sub_220EF30(v14) )
  {
    v15 = *(_QWORD *)(v14 + 40);
    v16 = v15 & 1;
    if ( (v15 & 1) == 0 )
      continue;
    v17 = *(_QWORD *)(v14 + 48);
    v82 = 1;
    v18 = *(_QWORD **)(v10 + 176);
    v81 = (__int64)&v81;
    v83 = v17;
    if ( v18 )
    {
      while ( 1 )
      {
        v19 = v18[6];
        v20 = (_QWORD *)v18[3];
        v21 = 0;
        if ( v17 < v19 )
        {
          v20 = (_QWORD *)v18[2];
          v21 = v15 & 1;
        }
        if ( !v20 )
          break;
        v18 = v20;
      }
      if ( !v21 )
      {
        if ( v17 > v19 )
          goto LABEL_13;
        goto LABEL_36;
      }
      if ( v18 == *(_QWORD **)(v10 + 184) )
      {
LABEL_13:
        v22 = 1;
        if ( v11 != v18 )
          goto LABEL_90;
        goto LABEL_14;
      }
    }
    else
    {
      v18 = v11;
      if ( v11 == *(_QWORD **)(v10 + 184) )
      {
        v22 = 1;
        goto LABEL_14;
      }
    }
    v76 = v15;
    v72 = v18;
    v68 = sub_220EF80(v18);
    v15 = v76;
    if ( v17 <= *(_QWORD *)(v68 + 48) )
    {
      v18 = (_QWORD *)v68;
    }
    else
    {
      v18 = v72;
      if ( v72 )
      {
        v22 = 1;
        if ( v11 != v72 )
LABEL_90:
          v22 = v17 < v18[6];
LABEL_14:
        v73 = v22;
        v77 = v18;
        v23 = (_QWORD *)sub_22077B0(56);
        v23[5] = 1;
        v24 = v23;
        v23[4] = v23 + 4;
        v23[6] = v83;
        sub_220F040(v73, v23, v77, v11);
        ++*(_QWORD *)(v10 + 200);
        v15 = *(_QWORD *)(v14 + 40);
        goto LABEL_15;
      }
    }
LABEL_36:
    v24 = v18;
LABEL_15:
    v25 = v24 + 4;
    if ( (v24[5] & 1) == 0 )
      v25 = 0;
    if ( (v15 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v26 = (_QWORD *)v10;
      v71 = v14;
      v27 = v15 & 0xFFFFFFFFFFFFFFFELL;
      v28 = v25;
      v29 = v16;
      v30 = v26;
      while ( 1 )
      {
        v31 = *(_QWORD *)(v27 + 16);
        v82 = 1;
        v32 = (_QWORD *)v30[22];
        v81 = (__int64)&v81;
        v83 = v31;
        if ( v32 )
        {
          while ( 1 )
          {
            v33 = v32[6];
            v34 = (_QWORD *)v32[3];
            v35 = 0;
            if ( v31 < v33 )
            {
              v34 = (_QWORD *)v32[2];
              v35 = v29;
            }
            if ( !v34 )
              break;
            v32 = v34;
          }
          if ( !v35 )
          {
            if ( v31 <= v33 )
              goto LABEL_38;
LABEL_27:
            v36 = 1;
            if ( v11 != v32 )
              goto LABEL_43;
            goto LABEL_28;
          }
          if ( (_QWORD *)v30[23] == v32 )
            goto LABEL_27;
        }
        else
        {
          v32 = v11;
          if ( (_QWORD *)v30[23] == v11 )
          {
            v36 = 1;
            goto LABEL_28;
          }
        }
        v79 = v31;
        v75 = v32;
        v40 = sub_220EF80(v32);
        v31 = v79;
        if ( v79 <= *(_QWORD *)(v40 + 48) )
        {
          v32 = (_QWORD *)v40;
LABEL_38:
          v37 = v32;
          goto LABEL_29;
        }
        v32 = v75;
        if ( !v75 )
          goto LABEL_38;
        v36 = 1;
        if ( v11 != v75 )
LABEL_43:
          v36 = v31 < v32[6];
LABEL_28:
        v74 = v36;
        v78 = v32;
        v37 = (_QWORD *)sub_22077B0(56);
        v37[4] = v37 + 4;
        v38 = v83;
        v37[5] = 1;
        v37[6] = v38;
        sub_220F040(v74, v37, v78, v11);
        ++v30[25];
LABEL_29:
        v39 = (unsigned __int64)(v37 + 4);
        if ( (v37[5] & 1) == 0 )
          v39 = 0;
        if ( v28 != (_QWORD *)v39 )
        {
          *(_QWORD *)(*v28 + 8LL) = v39 | *(_QWORD *)(*v28 + 8LL) & 1LL;
          *v28 = *(_QWORD *)v39;
          *(_QWORD *)(v39 + 8) &= ~1uLL;
          *(_QWORD *)v39 = v28;
        }
        v27 = *(_QWORD *)(v27 + 8) & 0xFFFFFFFFFFFFFFFELL;
        if ( !v27 )
        {
          v14 = v71;
          v10 = (__int64)v30;
          break;
        }
      }
    }
  }
  sub_1903F40(v85);
  sub_1905200(v10);
  v41 = *(_QWORD *)(v10 + 32);
  v42 = *(_QWORD *)(v10 + 40);
  v43 = v41;
  if ( v41 != v42 )
  {
    do
    {
      if ( *(_DWORD *)(v43 + 32) > 0x40u )
      {
        v44 = *(_QWORD *)(v43 + 24);
        if ( v44 )
          j_j___libc_free_0_0(v44);
      }
      if ( *(_DWORD *)(v43 + 16) > 0x40u )
      {
        v45 = *(_QWORD *)(v43 + 8);
        if ( v45 )
          j_j___libc_free_0_0(v45);
      }
      v43 += 40;
    }
    while ( v42 != v43 );
    *(_QWORD *)(v10 + 40) = v41;
  }
  v46 = *(_DWORD *)(v10 + 224);
  ++*(_QWORD *)(v10 + 208);
  if ( !v46 )
  {
    if ( !*(_DWORD *)(v10 + 228) )
      goto LABEL_61;
    v47 = *(unsigned int *)(v10 + 232);
    if ( (unsigned int)v47 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(v10 + 216));
      *(_QWORD *)(v10 + 216) = 0;
      *(_QWORD *)(v10 + 224) = 0;
      *(_DWORD *)(v10 + 232) = 0;
      goto LABEL_61;
    }
    goto LABEL_58;
  }
  v57 = 4 * v46;
  v47 = *(unsigned int *)(v10 + 232);
  if ( (unsigned int)(4 * v46) < 0x40 )
    v57 = 64;
  if ( (unsigned int)v47 <= v57 )
  {
LABEL_58:
    v48 = *(_QWORD **)(v10 + 216);
    for ( i = &v48[2 * v47]; i != v48; v48 += 2 )
      *v48 = -8;
    *(_QWORD *)(v10 + 224) = 0;
    goto LABEL_61;
  }
  v58 = *(_QWORD **)(v10 + 216);
  v59 = v46 - 1;
  if ( !v59 )
  {
    v64 = 2048;
    v63 = 128;
LABEL_79:
    j___libc_free_0(v58);
    *(_DWORD *)(v10 + 232) = v63;
    v65 = (_QWORD *)sub_22077B0(v64);
    v66 = *(unsigned int *)(v10 + 232);
    *(_QWORD *)(v10 + 224) = 0;
    *(_QWORD *)(v10 + 216) = v65;
    for ( j = &v65[2 * v66]; j != v65; v65 += 2 )
    {
      if ( v65 )
        *v65 = -8;
    }
    goto LABEL_61;
  }
  _BitScanReverse(&v59, v59);
  v60 = 1 << (33 - (v59 ^ 0x1F));
  if ( v60 < 64 )
    v60 = 64;
  if ( (_DWORD)v47 != v60 )
  {
    v61 = (4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1);
    v62 = ((v61 | (v61 >> 2)) >> 4) | v61 | (v61 >> 2) | ((((v61 | (v61 >> 2)) >> 4) | v61 | (v61 >> 2)) >> 8);
    v63 = (v62 | (v62 >> 16)) + 1;
    v64 = 16 * ((v62 | (v62 >> 16)) + 1);
    goto LABEL_79;
  }
  *(_QWORD *)(v10 + 224) = 0;
  v69 = &v58[2 * (unsigned int)v47];
  do
  {
    if ( v58 )
      *v58 = -8;
    v58 += 2;
  }
  while ( v69 != v58 );
LABEL_61:
  v50 = *(_QWORD *)(v10 + 240);
  if ( v50 != *(_QWORD *)(v10 + 248) )
    *(_QWORD *)(v10 + 248) = v50;
  ++*(_QWORD *)(v10 + 56);
  v51 = *(void **)(v10 + 72);
  if ( v51 == *(void **)(v10 + 64) )
    goto LABEL_68;
  v52 = 4 * (*(_DWORD *)(v10 + 84) - *(_DWORD *)(v10 + 88));
  v53 = *(unsigned int *)(v10 + 80);
  if ( v52 < 0x20 )
    v52 = 32;
  if ( (unsigned int)v53 <= v52 )
  {
    memset(v51, -1, 8 * v53);
LABEL_68:
    *(_QWORD *)(v10 + 84) = 0;
    goto LABEL_69;
  }
  sub_16CC920(v10 + 56);
LABEL_69:
  *(_QWORD *)(v10 + 264) = **(_QWORD **)(a2 + 40);
  sub_1904650(v10, a2, v10 + 56);
  sub_1906720(v10, v10 + 56);
  sub_1905CD0(v10);
  result = sub_19083A0(v10, a3, a4, a5, a6, v54, v55, a9, a10);
  if ( (_BYTE)result )
  {
    v80 = result;
    sub_1904910(v10);
    return v80;
  }
  return result;
}

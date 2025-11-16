// Function: sub_3142480
// Address: 0x3142480
//
__int64 __fastcall sub_3142480(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        unsigned __int8 *a5,
        size_t a6,
        unsigned __int8 *a7,
        size_t a8,
        unsigned __int8 *a9,
        size_t a10,
        __int64 a11)
{
  __int64 i; // rcx
  __int64 v15; // r14
  __int64 v16; // r12
  unsigned __int64 v17; // r8
  int v18; // r12d
  unsigned int v19; // eax
  size_t v20; // rdx
  int v21; // r9d
  unsigned __int64 v22; // rbx
  const void *v23; // rsi
  bool v24; // al
  int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 result; // rax
  __int64 v31; // rdi
  _QWORD *v32; // rax
  _WORD *v33; // rdi
  __int64 v34; // r12
  unsigned __int64 v35; // rax
  _WORD *v36; // rdi
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  _WORD *v39; // rdx
  __int64 v40; // r12
  _BYTE *v41; // rdi
  __int64 *v42; // rbx
  __int64 v43; // r14
  __int64 v44; // r13
  __int64 v45; // r12
  __int64 v46; // rsi
  int v47; // r11d
  unsigned int j; // eax
  unsigned int v49; // eax
  __int64 v50; // rax
  void (__fastcall *v51)(__int64, unsigned int *, unsigned __int64, _QWORD *, unsigned __int64, unsigned __int64); // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  unsigned int v58; // ecx
  size_t v59; // [rsp+0h] [rbp-E0h]
  int v60; // [rsp+14h] [rbp-CCh]
  unsigned __int64 v63; // [rsp+48h] [rbp-98h]
  unsigned int v64; // [rsp+48h] [rbp-98h]
  unsigned __int64 v65; // [rsp+48h] [rbp-98h]
  size_t nb; // [rsp+58h] [rbp-88h]
  size_t n; // [rsp+58h] [rbp-88h]
  size_t na; // [rsp+58h] [rbp-88h]
  unsigned int v70; // [rsp+6Ch] [rbp-74h] BYREF
  _QWORD v71[4]; // [rsp+70h] [rbp-70h] BYREF
  _QWORD v72[10]; // [rsp+90h] [rbp-50h] BYREF

  i = *(_QWORD *)(a1 + 96);
  v70 = 0;
  v15 = i + 32LL * *(unsigned int *)(a1 + 104) - 32;
  v16 = *(unsigned int *)(v15 + 24);
  v17 = *(_QWORD *)(v15 + 8);
  if ( !(_DWORD)v16 )
  {
    v28 = *(_QWORD *)(v15 + 8);
    v27 = 0;
LABEL_19:
    v22 = v17 + 48 * v16;
    goto LABEL_13;
  }
  v18 = v16 - 1;
  v63 = *(_QWORD *)(v15 + 8);
  nb = a4;
  v19 = sub_C94890(a3, a4);
  v17 = v63;
  v20 = nb;
  v21 = 1;
  for ( i = v18 & v19; ; i = v18 & v58 )
  {
    v22 = v17 + 48LL * (unsigned int)i;
    v23 = *(const void **)v22;
    if ( *(_QWORD *)v22 == -1 )
      break;
    v24 = (_QWORD *)((char *)a3 + 2) == 0;
    if ( v23 != (const void *)-2LL )
    {
      if ( *(_QWORD *)(v22 + 8) != v20 )
        goto LABEL_81;
      v60 = v21;
      v64 = i;
      n = v17;
      if ( !v20 )
        goto LABEL_12;
      v59 = v20;
      v25 = memcmp(a3, v23, v20);
      v20 = v59;
      v17 = n;
      i = v64;
      v21 = v60;
      v24 = v25 == 0;
    }
    if ( v24 )
      goto LABEL_12;
    if ( v23 == (const void *)-1LL )
      goto LABEL_10;
LABEL_81:
    v58 = v21 + i;
    ++v21;
  }
  if ( a3 != (_QWORD *)-1LL )
  {
LABEL_10:
    v17 = *(_QWORD *)(v15 + 8);
    v16 = *(unsigned int *)(v15 + 24);
    v26 = *(_QWORD *)(a1 + 96) + 32LL * *(unsigned int *)(a1 + 104) - 32;
    v27 = *(unsigned int *)(v26 + 24);
    v28 = *(_QWORD *)(v26 + 8);
    goto LABEL_19;
  }
LABEL_12:
  v29 = *(_QWORD *)(a1 + 96) + 32LL * *(unsigned int *)(a1 + 104) - 32;
  v27 = *(unsigned int *)(v29 + 24);
  v28 = *(_QWORD *)(v29 + 8);
LABEL_13:
  result = v28 + 48 * v27;
  if ( v22 == result )
    return result;
  v65 = v22 + 16;
  result = a2;
  v31 = *(unsigned int *)(a2 + 16);
  if ( !(_DWORD)v31 )
    goto LABEL_15;
  v42 = *(__int64 **)(a2 + 8);
  result = 3LL * *(unsigned int *)(a2 + 24);
  v17 = (unsigned __int64)&v42[3 * *(unsigned int *)(a2 + 24)];
  if ( v42 == (__int64 *)v17 )
    goto LABEL_15;
  while ( 2 )
  {
    result = v42[2];
    if ( result != -4096 )
    {
      if ( result != -8192 || v42[1] != -8192 || *v42 != -8192 )
        break;
      goto LABEL_73;
    }
    if ( v42[1] == -4096 && *v42 == -4096 )
    {
LABEL_73:
      v42 += 3;
      if ( (__int64 *)v17 == v42 )
        goto LABEL_15;
      continue;
    }
    break;
  }
  if ( (__int64 *)v17 != v42 )
  {
    na = v17;
LABEL_43:
    v43 = *v42;
    v44 = v42[1];
    v45 = v42[2];
    v28 = *(unsigned int *)(a2 + 56);
    v46 = *(_QWORD *)(a2 + 40);
    if ( !(_DWORD)v28 )
      goto LABEL_51;
    v47 = 1;
    v17 = (0xBF58476D1CE4E5B9LL
         * ((969526130LL * (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4))) & 0xFFFFFFFELL
          | ((unsigned __int64)(((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4)) << 32))) >> 31;
    for ( j = (v28 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * ((unsigned int)v17 ^ (-279380126 * (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4)))
                | ((unsigned __int64)(((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4)) << 32))) >> 31)
             ^ (484763065 * (v17 ^ (-279380126 * (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4))))));
          ;
          j = (v28 - 1) & v49 )
    {
      i = v46 + 24LL * j;
      v31 = *(_QWORD *)(i + 16);
      if ( v45 == v31 && v44 == *(_QWORD *)(i + 8) && *(_QWORD *)i == v43 )
        break;
      if ( v31 == -4096 && *(_QWORD *)(i + 8) == -4096 && *(_QWORD *)i == -4096 )
        goto LABEL_51;
      v49 = v47 + j;
      ++v47;
    }
    result = v46 + 24 * v28;
    if ( i == result )
    {
LABEL_51:
      v50 = *(_QWORD *)a1;
      v71[0] = *v42;
      v51 = *(void (__fastcall **)(__int64, unsigned int *, unsigned __int64, _QWORD *, unsigned __int64, unsigned __int64))(v50 + 16);
      v71[1] = v44;
      v71[2] = v45;
      v51(a1, &v70, v65, v71, v17, 0xBF58476D1CE4E5B9LL);
      v31 = a1;
      v72[0] = v43;
      v72[1] = v44;
      v72[2] = v45;
      result = sub_31420E0(a1, v72, a11);
    }
    for ( v42 += 3; (__int64 *)na != v42; v42 += 3 )
    {
      result = v42[2];
      if ( result == -4096 )
      {
        if ( v42[1] != -4096 || *v42 != -4096 )
        {
LABEL_55:
          if ( (__int64 *)na == v42 )
            break;
          goto LABEL_43;
        }
      }
      else if ( result != -8192 || v42[1] != -8192 || *v42 != -8192 )
      {
        goto LABEL_55;
      }
    }
  }
LABEL_15:
  if ( v70 )
  {
    v32 = sub_CB7210(v31, v70, v28, i, v17);
    v33 = (_WORD *)v32[4];
    v34 = (__int64)v32;
    v35 = v32[3] - (_QWORD)v33;
    if ( v35 < a10 )
    {
      v52 = sub_CB6200(v34, a9, a10);
      v33 = *(_WORD **)(v52 + 32);
      v34 = v52;
      v35 = *(_QWORD *)(v52 + 24) - (_QWORD)v33;
    }
    else if ( a10 )
    {
      memcpy(v33, a9, a10);
      v56 = *(_QWORD *)(v34 + 24);
      v33 = (_WORD *)(a10 + *(_QWORD *)(v34 + 32));
      *(_QWORD *)(v34 + 32) = v33;
      v35 = v56 - (_QWORD)v33;
    }
    if ( v35 <= 1 )
    {
      v55 = sub_CB6200(v34, (unsigned __int8 *)", ", 2u);
      v36 = *(_WORD **)(v55 + 32);
      v34 = v55;
    }
    else
    {
      *v33 = 8236;
      v36 = (_WORD *)(*(_QWORD *)(v34 + 32) + 2LL);
      *(_QWORD *)(v34 + 32) = v36;
    }
    v37 = *(_QWORD *)(v34 + 24) - (_QWORD)v36;
    if ( v37 < a6 )
    {
      v54 = sub_CB6200(v34, a5, a6);
      v36 = *(_WORD **)(v54 + 32);
      v34 = v54;
      v37 = *(_QWORD *)(v54 + 24) - (_QWORD)v36;
    }
    else if ( a6 )
    {
      memcpy(v36, a5, a6);
      v57 = *(_QWORD *)(v34 + 24);
      v36 = (_WORD *)(a6 + *(_QWORD *)(v34 + 32));
      *(_QWORD *)(v34 + 32) = v36;
      v37 = v57 - (_QWORD)v36;
    }
    if ( v37 <= 1 )
    {
      v34 = sub_CB6200(v34, (unsigned __int8 *)", ", 2u);
    }
    else
    {
      *v36 = 8236;
      *(_QWORD *)(v34 + 32) += 2LL;
    }
    v38 = sub_CB59D0(v34, v70);
    v39 = *(_WORD **)(v38 + 32);
    v40 = v38;
    if ( *(_QWORD *)(v38 + 24) - (_QWORD)v39 <= 1u )
    {
      v53 = sub_CB6200(v38, (unsigned __int8 *)", ", 2u);
      v41 = *(_BYTE **)(v53 + 32);
      v40 = v53;
    }
    else
    {
      *v39 = 8236;
      v41 = (_BYTE *)(*(_QWORD *)(v38 + 32) + 2LL);
      *(_QWORD *)(v38 + 32) = v41;
    }
    result = *(_QWORD *)(v40 + 24);
    if ( result - (__int64)v41 < a8 )
    {
      v40 = sub_CB6200(v40, a7, a8);
      result = *(_QWORD *)(v40 + 24);
      v41 = *(_BYTE **)(v40 + 32);
    }
    else if ( a8 )
    {
      memcpy(v41, a7, a8);
      result = *(_QWORD *)(v40 + 24);
      v41 = (_BYTE *)(a8 + *(_QWORD *)(v40 + 32));
      *(_QWORD *)(v40 + 32) = v41;
    }
    if ( (_BYTE *)result == v41 )
    {
      result = sub_CB6200(v40, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v41 = 10;
      ++*(_QWORD *)(v40 + 32);
    }
    *(_BYTE *)(a1 + 144) = 1;
  }
  else
  {
    *(_BYTE *)(a1 + 144) = 0;
  }
  return result;
}

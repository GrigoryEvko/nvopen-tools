// Function: sub_1D15B50
// Address: 0x1d15b50
//
__int64 __fastcall sub_1D15B50(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, char a5, int a6)
{
  __int64 v7; // r14
  _QWORD *v8; // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // rbx
  __int64 v11; // rax
  int v13; // ebx
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v18; // r14
  __int64 v19; // rcx
  __int64 v20; // rbx
  __int64 *v21; // r13
  __int64 *v22; // rbx
  __int64 v23; // r12
  __int64 *v24; // rax
  char v25; // dl
  __int64 v26; // rdi
  int v27; // ebx
  void *v28; // r13
  size_t v29; // r12
  char v30; // di
  __int64 *v31; // rsi
  unsigned int v32; // edi
  __int64 *v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rax
  __int64 v39; // rbx
  __int64 v40; // rt0
  _QWORD *v41; // rdx
  const void *v42; // [rsp+10h] [rbp-A0h]
  int v43; // [rsp+1Ch] [rbp-94h]
  unsigned __int8 v46; // [rsp+27h] [rbp-89h]
  void *src; // [rsp+30h] [rbp-80h] BYREF
  __int64 v49; // [rsp+38h] [rbp-78h]
  _BYTE v50[112]; // [rsp+40h] [rbp-70h] BYREF

  v7 = a2;
  src = v50;
  v8 = *(_QWORD **)(a2 + 16);
  v49 = 0x800000000LL;
  v9 = *(_QWORD **)(a2 + 8);
  if ( v8 == v9 )
  {
    v10 = &v9[*(unsigned int *)(a2 + 28)];
    if ( v9 == v10 )
    {
      v41 = *(_QWORD **)(a2 + 8);
    }
    else
    {
      do
      {
        if ( a1 == *v9 )
          break;
        ++v9;
      }
      while ( v10 != v9 );
      v41 = v10;
    }
    goto LABEL_62;
  }
  v10 = &v8[*(unsigned int *)(a2 + 24)];
  v9 = sub_16CC9F0(a2, a1);
  if ( a1 == *v9 )
  {
    v36 = *(_QWORD *)(a2 + 16);
    if ( v36 == *(_QWORD *)(a2 + 8) )
      v37 = *(unsigned int *)(a2 + 28);
    else
      v37 = *(unsigned int *)(a2 + 24);
    v41 = (_QWORD *)(v36 + 8 * v37);
    goto LABEL_62;
  }
  v11 = *(_QWORD *)(a2 + 16);
  if ( v11 == *(_QWORD *)(a2 + 8) )
  {
    v9 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(a2 + 28));
    v41 = v9;
LABEL_62:
    while ( v41 != v9 && *v9 >= 0xFFFFFFFFFFFFFFFELL )
      ++v9;
    goto LABEL_5;
  }
  v9 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(a2 + 24));
LABEL_5:
  if ( v9 != v10 )
  {
    v46 = 1;
    goto LABEL_7;
  }
  v13 = *(_DWORD *)(a1 + 28);
  v14 = ~v13;
  if ( v13 >= -1 )
    v14 = *(_DWORD *)(a1 + 28);
  v42 = (const void *)(a3 + 16);
  v43 = v14;
  v15 = *(unsigned int *)(a3 + 8);
  if ( !(_DWORD)v15 )
  {
LABEL_70:
    v46 = 0;
    goto LABEL_29;
  }
  v16 = a3;
  v18 = v16;
  while ( 1 )
  {
    v19 = (unsigned int)v15;
    v15 = (unsigned int)(v15 - 1);
    v20 = *(_QWORD *)(*(_QWORD *)v18 + 8 * v19 - 8);
    *(_DWORD *)(v18 + 8) = v15;
    if ( a5 && *(_WORD *)(v20 + 24) != 2 && v43 > *(_DWORD *)(v20 + 28) && *(_DWORD *)(v20 + 28) > 0 && v43 > 0 )
    {
      v35 = (unsigned int)v49;
      if ( (unsigned int)v49 >= HIDWORD(v49) )
      {
        sub_16CD150((__int64)&src, v50, 0, 8, a5, a6);
        v35 = (unsigned int)v49;
      }
      *((_QWORD *)src + v35) = v20;
      v15 = *(unsigned int *)(v18 + 8);
      LODWORD(v49) = v49 + 1;
      goto LABEL_27;
    }
    v21 = *(__int64 **)(v20 + 32);
    v22 = &v21[5 * *(unsigned int *)(v20 + 56)];
    if ( v21 != v22 )
      break;
LABEL_25:
    if ( a4 && a4 <= *(_DWORD *)(a2 + 28) - *(_DWORD *)(a2 + 32) )
    {
      v39 = v18;
      v7 = a2;
      a3 = v39;
      goto LABEL_70;
    }
LABEL_27:
    if ( !(_DWORD)v15 )
    {
      v46 = 0;
      v26 = v18;
      v7 = a2;
      a3 = v26;
      goto LABEL_29;
    }
  }
  v46 = 0;
  do
  {
    while ( 1 )
    {
      v23 = *v21;
      v24 = *(__int64 **)(a2 + 8);
      if ( *(__int64 **)(a2 + 16) != v24 )
        goto LABEL_21;
      v31 = &v24[*(unsigned int *)(a2 + 28)];
      v32 = *(_DWORD *)(a2 + 28);
      if ( v24 == v31 )
        break;
      v33 = 0;
      while ( v23 != *v24 )
      {
        if ( *v24 == -2 )
          v33 = v24;
        if ( v31 == ++v24 )
        {
          if ( !v33 )
            goto LABEL_49;
          *v33 = v23;
          --*(_DWORD *)(a2 + 32);
          ++*(_QWORD *)a2;
          goto LABEL_45;
        }
      }
LABEL_22:
      v21 += 5;
      if ( a1 != v23 )
        goto LABEL_23;
LABEL_47:
      if ( v22 == v21 )
      {
        v38 = v18;
        v46 = 1;
        v7 = a2;
        a3 = v38;
        v15 = *(unsigned int *)(v38 + 8);
        goto LABEL_29;
      }
      v46 = 1;
    }
LABEL_49:
    if ( v32 < *(_DWORD *)(a2 + 24) )
    {
      *(_DWORD *)(a2 + 28) = v32 + 1;
      *v31 = v23;
      ++*(_QWORD *)a2;
      v34 = *(unsigned int *)(v18 + 8);
      if ( (unsigned int)v34 < *(_DWORD *)(v18 + 12) )
        goto LABEL_46;
LABEL_51:
      sub_16CD150(v18, v42, 0, 8, a5, a6);
      v34 = *(unsigned int *)(v18 + 8);
      goto LABEL_46;
    }
LABEL_21:
    sub_16CCBA0(a2, *v21);
    if ( !v25 )
      goto LABEL_22;
LABEL_45:
    v34 = *(unsigned int *)(v18 + 8);
    if ( (unsigned int)v34 >= *(_DWORD *)(v18 + 12) )
      goto LABEL_51;
LABEL_46:
    v21 += 5;
    *(_QWORD *)(*(_QWORD *)v18 + 8 * v34) = v23;
    ++*(_DWORD *)(v18 + 8);
    if ( a1 == v23 )
      goto LABEL_47;
LABEL_23:
    ;
  }
  while ( v22 != v21 );
  v15 = *(unsigned int *)(v18 + 8);
  if ( !v46 )
    goto LABEL_25;
  v40 = a2;
  a3 = v18;
  v7 = v40;
LABEL_29:
  v27 = v49;
  v28 = src;
  v29 = 8LL * (unsigned int)v49;
  if ( (unsigned int)v49 > (unsigned __int64)*(unsigned int *)(a3 + 12) - v15 )
  {
    sub_16CD150(a3, (const void *)(a3 + 16), (unsigned int)v49 + v15, 8, a5, a6);
    v15 = *(unsigned int *)(a3 + 8);
  }
  if ( v29 )
  {
    memcpy((void *)(*(_QWORD *)a3 + 8 * v15), v28, v29);
    LODWORD(v15) = *(_DWORD *)(a3 + 8);
  }
  *(_DWORD *)(a3 + 8) = v27 + v15;
  if ( a4 )
  {
    v30 = v46;
    if ( a4 <= *(_DWORD *)(v7 + 28) - *(_DWORD *)(v7 + 32) )
      v30 = 1;
    v46 = v30;
  }
LABEL_7:
  if ( src != v50 )
    _libc_free((unsigned __int64)src);
  return v46;
}

// Function: sub_2F90200
// Address: 0x2f90200
//
void __fastcall sub_2F90200(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v7; // rax
  __int64 v8; // r12
  __int64 v9; // r12
  int v10; // r14d
  char *v11; // rax
  char *v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  char *v22; // rsi
  unsigned __int64 *v23; // r8
  unsigned __int64 v24; // rbx
  __int64 i; // r13
  int v26; // eax
  char *v27; // rsi
  char *v28; // rax
  int v29; // r13d
  __int64 v30; // rbx
  unsigned int v31; // esi
  __int64 v32; // rcx
  _QWORD *v33; // rbx
  _QWORD *v34; // rcx
  __int64 v35; // rax
  _DWORD *v36; // rdx
  char *v38; // rsi
  int v39; // ecx
  unsigned __int64 v40; // rax
  unsigned int v41; // r12d
  int v42; // r14d
  __int64 v43; // rdx
  unsigned __int64 v44; // rbx
  void *v45; // rdi
  char *v46; // [rsp+8h] [rbp-68h]
  _QWORD *v47; // [rsp+8h] [rbp-68h]
  unsigned __int64 *v48; // [rsp+8h] [rbp-68h]
  unsigned __int64 v49; // [rsp+18h] [rbp-58h] BYREF
  void *src; // [rsp+20h] [rbp-50h] BYREF
  char *v51; // [rsp+28h] [rbp-48h]
  char *v52; // [rsp+30h] [rbp-40h]

  v7 = *(unsigned __int64 **)a1;
  *(_BYTE *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 32) = 0;
  v8 = v7[1] - *v7;
  src = 0;
  v9 = v8 >> 8;
  v51 = 0;
  v52 = 0;
  LOBYTE(v10) = v9;
  if ( !(_DWORD)v9 )
  {
    v13 = *(_QWORD *)(a1 + 304);
    v14 = *(_QWORD *)(a1 + 296);
    if ( v14 == v13 )
    {
      v16 = *(_QWORD *)(a1 + 328);
      v17 = *(_QWORD *)(a1 + 320);
      v19 = (v16 - v17) >> 2;
      goto LABEL_7;
    }
LABEL_47:
    v43 = v14 + 4LL * (unsigned int)v9;
    if ( v43 != v13 )
      *(_QWORD *)(a1 + 304) = v43;
    goto LABEL_6;
  }
  v11 = (char *)sub_22077B0(8LL * (unsigned int)v9);
  v12 = v11;
  if ( v51 - (_BYTE *)src > 0 )
  {
    v46 = (char *)memmove(v11, src, v51 - (_BYTE *)src);
    j_j___libc_free_0((unsigned __int64)src);
    v12 = v46;
  }
  v13 = *(_QWORD *)(a1 + 304);
  src = v12;
  v51 = v12;
  v14 = *(_QWORD *)(a1 + 296);
  v52 = &v12[8 * (unsigned int)v9];
  v15 = (v13 - v14) >> 2;
  if ( (unsigned int)v9 > v15 )
  {
    sub_1F025F0(a1 + 296, (unsigned int)v9 - v15);
    goto LABEL_6;
  }
  if ( (unsigned int)v9 < v15 )
    goto LABEL_47;
LABEL_6:
  v16 = *(_QWORD *)(a1 + 328);
  v17 = *(_QWORD *)(a1 + 320);
  v18 = (v16 - v17) >> 2;
  v19 = v18;
  if ( (unsigned int)v9 > v18 )
  {
    sub_1F025F0(a1 + 320, (unsigned int)v9 - v18);
    goto LABEL_10;
  }
LABEL_7:
  if ( (unsigned int)v9 < v19 )
  {
    v20 = v17 + 4LL * (unsigned int)v9;
    if ( v20 != v16 )
      *(_QWORD *)(a1 + 328) = v20;
  }
LABEL_10:
  v21 = *(_QWORD *)(a1 + 8);
  if ( v21 )
  {
    v22 = v51;
    if ( v51 == v52 )
    {
      sub_2ECAD30((__int64)&src, v51, (_QWORD *)(a1 + 8));
    }
    else
    {
      if ( v51 )
      {
        *(_QWORD *)v51 = v21;
        v22 = v51;
      }
      v51 = v22 + 8;
    }
  }
  v23 = &v49;
  v24 = **(_QWORD **)a1;
  for ( i = *(_QWORD *)(*(_QWORD *)a1 + 8LL); i != v24; v51 = v27 + 8 )
  {
    while ( 1 )
    {
      v26 = *(_DWORD *)(v24 + 128);
      *(_DWORD *)(*(_QWORD *)(a1 + 320) + 4LL * *(int *)(v24 + 200)) = v26;
      if ( !v26 )
        break;
LABEL_17:
      v24 += 256LL;
      if ( i == v24 )
        goto LABEL_23;
    }
    v49 = v24;
    v27 = v51;
    if ( v51 == v52 )
    {
      v48 = v23;
      sub_2F3A320((__int64)&src, v51, v23);
      v23 = v48;
      goto LABEL_17;
    }
    if ( v51 )
    {
      *(_QWORD *)v51 = v24;
      v27 = v51;
    }
    v24 += 256LL;
  }
LABEL_23:
  v28 = v51;
  v29 = v9;
  if ( v51 != src )
  {
    while ( 1 )
    {
      v30 = *((_QWORD *)v28 - 1);
      v51 = v28 - 8;
      v31 = *(_DWORD *)(v30 + 200);
      if ( v31 < (unsigned int)v9 )
        sub_2F8FC70(a1, v31, --v29);
      v32 = 2LL * *(unsigned int *)(v30 + 48);
      v33 = *(_QWORD **)(v30 + 40);
      v34 = &v33[v32];
      if ( v34 != v33 )
        break;
LABEL_35:
      v28 = v51;
      if ( v51 == src )
        goto LABEL_36;
    }
    while ( 1 )
    {
      v49 = *v33 & 0xFFFFFFFFFFFFFFF8LL;
      v35 = *(unsigned int *)(v49 + 200);
      if ( (unsigned int)v35 >= (unsigned int)v9 )
        goto LABEL_28;
      v36 = (_DWORD *)(*(_QWORD *)(a1 + 320) + 4 * v35);
      if ( (*v36)-- != 1 )
        goto LABEL_28;
      v38 = v51;
      if ( v51 == v52 )
      {
        v47 = v34;
        sub_2ECAD30((__int64)&src, v51, &v49);
        v34 = v47;
LABEL_28:
        v33 += 2;
        if ( v34 == v33 )
          goto LABEL_35;
      }
      else
      {
        if ( v51 )
        {
          *(_QWORD *)v51 = v49;
          v38 = v51;
        }
        v33 += 2;
        v51 = v38 + 8;
        if ( v34 == v33 )
          goto LABEL_35;
      }
    }
  }
LABEL_36:
  v39 = *(_DWORD *)(a1 + 408) & 0x3F;
  if ( v39 )
    *(_QWORD *)(*(_QWORD *)(a1 + 344) + 8LL * *(unsigned int *)(a1 + 352) - 8) &= ~(-1LL << v39);
  *(_DWORD *)(a1 + 408) = v9;
  v40 = *(unsigned int *)(a1 + 352);
  v41 = (unsigned int)(v9 + 63) >> 6;
  if ( v41 != v40 )
  {
    if ( v41 >= v40 )
    {
      v44 = v41 - v40;
      if ( v41 > (unsigned __int64)*(unsigned int *)(a1 + 356) )
      {
        sub_C8D5F0(a1 + 344, (const void *)(a1 + 360), v41, 8u, (__int64)v23, a6);
        v40 = *(unsigned int *)(a1 + 352);
      }
      v45 = (void *)(*(_QWORD *)(a1 + 344) + 8 * v40);
      if ( 8 * v44 )
      {
        memset(v45, 0, 8 * v44);
        LODWORD(v40) = *(_DWORD *)(a1 + 352);
      }
      v10 = *(_DWORD *)(a1 + 408);
      *(_DWORD *)(a1 + 352) = v44 + v40;
    }
    else
    {
      *(_DWORD *)(a1 + 352) = v41;
    }
  }
  v42 = v10 & 0x3F;
  if ( v42 )
    *(_QWORD *)(*(_QWORD *)(a1 + 344) + 8LL * *(unsigned int *)(a1 + 352) - 8) &= ~(-1LL << v42);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
}

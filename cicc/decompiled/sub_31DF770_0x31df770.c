// Function: sub_31DF770
// Address: 0x31df770
//
__int64 __fastcall sub_31DF770(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rcx
  __int64 v4; // r15
  __int64 *v5; // r12
  unsigned __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  char *v12; // r8
  __int64 v13; // r9
  __int64 v14; // r11
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r9
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rsi
  char *v24; // r13
  __int64 v25; // r10
  __int64 v26; // r11
  __int64 v27; // rcx
  unsigned int v28; // edx
  void *v29; // rdi
  __int64 v30; // r10
  size_t v31; // rax
  char *v32; // r15
  __int64 v33; // r11
  __int64 v34; // [rsp+8h] [rbp-58h]
  int v35; // [rsp+10h] [rbp-50h]
  size_t v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  char *v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+20h] [rbp-40h]
  int v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  int v44; // [rsp+28h] [rbp-38h]
  __int64 v45; // [rsp+28h] [rbp-38h]
  char *v46; // [rsp+28h] [rbp-38h]
  __int64 v47; // [rsp+28h] [rbp-38h]
  __int64 v48; // [rsp+28h] [rbp-38h]
  char *v49; // [rsp+28h] [rbp-38h]
  unsigned __int64 v50; // [rsp+28h] [rbp-38h]

  result = *a2;
  v3 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*a2 & 4) != 0 )
  {
    v5 = *(__int64 **)v3;
    result = *(unsigned int *)(v3 + 8);
    v4 = *(_QWORD *)v3 + 8 * result;
  }
  else
  {
    v4 = (__int64)(a2 + 1);
    v5 = a2;
    if ( !v3 )
      v4 = (__int64)a2;
  }
  v6 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v7 = (*a1 >> 2) & 1;
  if ( ((*a1 >> 2) & 1) != 0 )
  {
    v10 = *(unsigned int *)(v6 + 8);
    if ( v5 == (__int64 *)v4 )
      return result;
    v7 = 8 * v10;
    if ( v6 )
    {
      v11 = *(unsigned int *)(v6 + 12);
      v12 = (char *)(*(_QWORD *)v6 + 8 * v10);
      v13 = v4 - (_QWORD)v5;
      v14 = (v4 - (__int64)v5) >> 3;
      v15 = v10 + v14;
      goto LABEL_15;
    }
    goto LABEL_7;
  }
  if ( !v6 )
  {
    if ( v5 == (__int64 *)v4 )
      return result;
LABEL_7:
    if ( (__int64 *)v4 == v5 + 1 )
    {
      result = *v5 & 0xFFFFFFFFFFFFFFFBLL;
      *a1 = result;
      return result;
    }
    v8 = sub_22077B0(0x30u);
    if ( v8 )
    {
      *(_QWORD *)v8 = v8 + 16;
      *(_QWORD *)(v8 + 8) = 0x400000000LL;
    }
    v9 = v8 | 4;
    *a1 = v8 | 4;
    goto LABEL_25;
  }
  if ( v5 == (__int64 *)v4 )
    return result;
  v7 = 8;
  v16 = sub_22077B0(0x30u);
  if ( v16 )
  {
    *(_QWORD *)v16 = v16 + 16;
    *(_QWORD *)(v16 + 8) = 0x400000000LL;
  }
  v18 = v16;
  v19 = v16 & 0xFFFFFFFFFFFFFFF8LL;
  v20 = *(unsigned int *)(v19 + 12);
  *a1 = v18 | 4;
  v21 = *(unsigned int *)(v19 + 8);
  if ( v21 + 1 > v20 )
  {
    v50 = v19;
    sub_C8D5F0(v19, (const void *)(v19 + 16), v21 + 1, 8u, v21 + 1, v17);
    v19 = v50;
    v21 = *(unsigned int *)(v50 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v19 + 8 * v21) = v6;
  v9 = *a1;
  ++*(_DWORD *)(v19 + 8);
LABEL_25:
  v6 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  v22 = *(unsigned int *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 8);
  v23 = *(_QWORD *)(v9 & 0xFFFFFFFFFFFFFFF8LL);
  result = 8 * v22;
  v12 = (char *)(v23 + 8 * v22);
  if ( (v9 & 4) != 0 )
  {
    v24 = (char *)(v23 + v7);
  }
  else
  {
    v24 = (char *)a1 + v7;
    v7 = (__int64)a1 + v7 - v23;
  }
  v11 = *(unsigned int *)(v6 + 12);
  v13 = v4 - (_QWORD)v5;
  v25 = (v4 - (__int64)v5) >> 3;
  LODWORD(v14) = v25;
  v15 = v25 + v22;
  if ( v24 != v12 )
  {
    if ( v15 > v11 )
    {
      sub_C8D5F0(v6, (const void *)(v6 + 16), v25 + v22, 8u, (__int64)v12, v13);
      v22 = *(unsigned int *)(v6 + 8);
      v23 = *(_QWORD *)v6;
      v25 = (v4 - (__int64)v5) >> 3;
      v13 = v4 - (_QWORD)v5;
      result = 8 * v22;
      v24 = (char *)(*(_QWORD *)v6 + v7);
      v12 = (char *)(*(_QWORD *)v6 + 8 * v22);
    }
    v26 = result - v7;
    v27 = (result - v7) >> 3;
    if ( result - v7 >= (unsigned __int64)v13 )
    {
      v29 = v12;
      v30 = result - v13;
      v31 = v13;
      v32 = (char *)(v23 + v30);
      v33 = v13 >> 3;
      if ( (v13 >> 3) + v22 > (unsigned __int64)*(unsigned int *)(v6 + 12) )
      {
        v34 = v13 >> 3;
        v36 = v13;
        v39 = v13;
        v43 = v30;
        v49 = v12;
        sub_C8D5F0(v6, (const void *)(v6 + 16), (v13 >> 3) + v22, 8u, (__int64)v12, v13);
        v22 = *(unsigned int *)(v6 + 8);
        LODWORD(v33) = v34;
        v31 = v36;
        v13 = v39;
        v30 = v43;
        v29 = (void *)(*(_QWORD *)v6 + 8 * v22);
        v12 = v49;
      }
      if ( v32 != v12 )
      {
        v35 = v33;
        v38 = v13;
        v41 = v30;
        v46 = v12;
        memmove(v29, v32, v31);
        LODWORD(v22) = *(_DWORD *)(v6 + 8);
        LODWORD(v33) = v35;
        v13 = v38;
        v30 = v41;
        v12 = v46;
      }
      *(_DWORD *)(v6 + 8) = v33 + v22;
      if ( v32 != v24 )
      {
        v47 = v13;
        memmove(&v12[-(v30 - v7)], v24, v30 - v7);
        v13 = v47;
      }
      return (__int64)memmove(v24, v5, v13);
    }
    else
    {
      v28 = v25 + v22;
      *(_DWORD *)(v6 + 8) = v28;
      if ( v24 != v12 )
      {
        v37 = (result - v7) >> 3;
        v40 = v12;
        v45 = result - v7;
        result = (__int64)memcpy((void *)(v23 + 8LL * v28 - v26), v24, result - v7);
        v27 = v37;
        v12 = v40;
        v26 = v45;
      }
      if ( v27 )
      {
        for ( result = 0; result != v27; ++result )
          *(_QWORD *)&v24[8 * result] = v5[result];
        v5 = (__int64 *)((char *)v5 + v26);
      }
      if ( v5 != (__int64 *)v4 )
        return (__int64)memcpy(v12, v5, v4 - (_QWORD)v5);
    }
    return result;
  }
LABEL_15:
  if ( v11 < v15 )
  {
    v42 = v14;
    v48 = v13;
    sub_C8D5F0(v6, (const void *)(v6 + 16), v15, 8u, (__int64)v12, v13);
    LODWORD(v14) = v42;
    v13 = v48;
    v12 = (char *)(*(_QWORD *)v6 + 8LL * *(unsigned int *)(v6 + 8));
  }
  v44 = v14;
  result = (__int64)memcpy(v12, v5, v13);
  *(_DWORD *)(v6 + 8) += v44;
  return result;
}

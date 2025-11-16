// Function: sub_1F02930
// Address: 0x1f02930
//
unsigned __int64 __fastcall sub_1F02930(unsigned __int64 **a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v5; // r12
  unsigned __int64 v6; // r12
  int v7; // r15d
  char *v8; // rax
  char *v9; // r14
  unsigned __int64 *v10; // rax
  unsigned __int64 *v11; // rdx
  unsigned __int64 v12; // rcx
  unsigned __int64 *v13; // rcx
  unsigned __int64 *v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rsi
  unsigned __int64 *v17; // rax
  unsigned __int64 *v18; // rax
  char *v19; // rsi
  unsigned __int64 *v20; // r9
  unsigned __int64 v21; // rbx
  unsigned __int64 i; // r14
  int v23; // eax
  char *v24; // rsi
  char *v25; // rax
  int v26; // r14d
  __int64 v27; // rbx
  unsigned int v28; // esi
  __int64 v29; // rcx
  unsigned __int64 *v30; // rbx
  __int64 v31; // rax
  _DWORD *v32; // rdx
  char *v34; // rsi
  unsigned __int64 *v35; // r14
  unsigned __int64 result; // rax
  unsigned __int64 *v37; // rdx
  unsigned __int64 v38; // rbx
  __int64 v39; // r8
  int v40; // ecx
  unsigned int v41; // r9d
  int v42; // ecx
  char *v43; // rax
  unsigned __int64 v44; // rdx
  unsigned int v45; // ebx
  unsigned int v46; // r12d
  int v47; // r15d
  unsigned __int64 v48; // rdx
  unsigned __int64 *v49; // [rsp+10h] [rbp-70h]
  unsigned __int64 *v50; // [rsp+10h] [rbp-70h]
  unsigned int v51; // [rsp+18h] [rbp-68h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  unsigned __int64 v53; // [rsp+28h] [rbp-58h] BYREF
  void *src; // [rsp+30h] [rbp-50h] BYREF
  char *v55; // [rsp+38h] [rbp-48h]
  char *v56; // [rsp+40h] [rbp-40h]

  v5 = (*a1)[1] - **a1;
  src = 0;
  v55 = 0;
  v6 = 0xF0F0F0F0F0F0F0F1LL * (v5 >> 4);
  v56 = 0;
  LOBYTE(v7) = v6;
  if ( !(_DWORD)v6 )
  {
    v10 = a1[3];
    v11 = a1[2];
    if ( v11 == v10 )
    {
      v13 = a1[6];
      v14 = a1[5];
      v16 = ((char *)v13 - (char *)v14) >> 2;
      goto LABEL_7;
    }
LABEL_43:
    v37 = (unsigned __int64 *)((char *)v11 + 4 * (unsigned int)v6);
    if ( v37 != v10 )
      a1[3] = v37;
    goto LABEL_6;
  }
  v8 = (char *)sub_22077B0(8LL * (unsigned int)v6);
  v9 = v8;
  if ( v55 - (_BYTE *)src > 0 )
  {
    memmove(v8, src, v55 - (_BYTE *)src);
    j_j___libc_free_0(src, v56 - (_BYTE *)src);
  }
  v10 = a1[3];
  v11 = a1[2];
  src = v9;
  v55 = v9;
  v56 = &v9[8 * (unsigned int)v6];
  v12 = ((char *)v10 - (char *)v11) >> 2;
  if ( (unsigned int)v6 > v12 )
  {
    sub_1F025F0((__int64)(a1 + 2), (unsigned int)v6 - v12);
    goto LABEL_6;
  }
  if ( (unsigned int)v6 < v12 )
    goto LABEL_43;
LABEL_6:
  v13 = a1[6];
  v14 = a1[5];
  v15 = ((char *)v13 - (char *)v14) >> 2;
  v16 = v15;
  if ( (unsigned int)v6 > v15 )
  {
    sub_1F025F0((__int64)(a1 + 5), (unsigned int)v6 - v15);
    goto LABEL_10;
  }
LABEL_7:
  if ( (unsigned int)v6 < v16 )
  {
    v17 = (unsigned __int64 *)((char *)v14 + 4 * (unsigned int)v6);
    if ( v17 != v13 )
      a1[6] = v17;
  }
LABEL_10:
  v18 = a1[1];
  if ( v18 )
  {
    v19 = v55;
    if ( v55 == v56 )
    {
      sub_1CFD630((__int64)&src, v55, a1 + 1);
    }
    else
    {
      if ( v55 )
      {
        *(_QWORD *)v55 = v18;
        v19 = v55;
      }
      v55 = v19 + 8;
    }
  }
  v20 = &v53;
  v21 = **a1;
  for ( i = (*a1)[1]; i != v21; v55 = v24 + 8 )
  {
    while ( 1 )
    {
      v23 = *(_DWORD *)(v21 + 120);
      v13 = (unsigned __int64 *)*(int *)(v21 + 192);
      *((_DWORD *)a1[5] + (_QWORD)v13) = v23;
      if ( !v23 )
        break;
LABEL_17:
      v21 += 272LL;
      if ( i == v21 )
        goto LABEL_23;
    }
    v53 = v21;
    v24 = v55;
    if ( v55 == v56 )
    {
      v50 = v20;
      sub_1D12610((__int64)&src, v55, v20);
      v20 = v50;
      goto LABEL_17;
    }
    if ( v55 )
    {
      *(_QWORD *)v55 = v21;
      v24 = v55;
    }
    v21 += 272LL;
  }
LABEL_23:
  v25 = v55;
  v26 = v6;
  if ( v55 != src )
  {
    while ( 1 )
    {
      v27 = *((_QWORD *)v25 - 1);
      v55 = v25 - 8;
      v28 = *(_DWORD *)(v27 + 192);
      if ( v28 < (unsigned int)v6 )
        sub_1F02230((__int64)a1, v28, --v26);
      v29 = 2LL * *(unsigned int *)(v27 + 40);
      v30 = *(unsigned __int64 **)(v27 + 32);
      v13 = &v30[v29];
      if ( v30 != v13 )
        break;
LABEL_35:
      v25 = v55;
      if ( v55 == src )
        goto LABEL_36;
    }
    while ( 1 )
    {
      v53 = *v30 & 0xFFFFFFFFFFFFFFF8LL;
      v31 = *(unsigned int *)(v53 + 192);
      if ( (unsigned int)v31 >= (unsigned int)v6 )
        goto LABEL_28;
      v32 = (_DWORD *)a1[5] + v31;
      if ( (*v32)-- != 1 )
        goto LABEL_28;
      v34 = v55;
      if ( v55 == v56 )
      {
        v49 = v13;
        sub_1CFD630((__int64)&src, v55, &v53);
        v13 = v49;
LABEL_28:
        v30 += 2;
        if ( v13 == v30 )
          goto LABEL_35;
      }
      else
      {
        if ( v55 )
        {
          *(_QWORD *)v55 = v53;
          v34 = v55;
        }
        v30 += 2;
        v55 = v34 + 8;
        if ( v13 == v30 )
          goto LABEL_35;
      }
    }
  }
LABEL_36:
  v35 = a1[9];
  if ( (unsigned __int64)(unsigned int)v6 <= (_QWORD)v35 << 6 )
    goto LABEL_37;
  v38 = (unsigned int)(v6 + 63) >> 6;
  if ( v38 < 2 * (__int64)v35 )
    v38 = 2LL * (_QWORD)v35;
  v39 = (__int64)realloc((unsigned __int64)a1[8], 8 * v38, 8 * (int)v38, (int)v13, a5, (int)v20);
  if ( !v39 && (8 * v38 || (v39 = malloc(1u)) == 0) )
  {
    v52 = v39;
    sub_16BD1C0("Allocation failed", 1u);
    v39 = v52;
  }
  v40 = *((_DWORD *)a1 + 20);
  a1[8] = (unsigned __int64 *)v39;
  a1[9] = (unsigned __int64 *)v38;
  v41 = (unsigned int)(v40 + 63) >> 6;
  if ( v38 > v41 )
  {
    v51 = (unsigned int)(v40 + 63) >> 6;
    memset((void *)(v39 + 8LL * v41), 0, 8 * (v38 - v41));
    v40 = *((_DWORD *)a1 + 20);
    v39 = (__int64)a1[8];
    v41 = v51;
  }
  v42 = v40 & 0x3F;
  if ( v42 )
  {
    *(_QWORD *)(v39 + 8LL * (v41 - 1)) &= ~(-1LL << v42);
    v39 = (__int64)a1[8];
  }
  v43 = (char *)a1[9] - (unsigned int)v35;
  if ( v43 )
  {
    memset((void *)(v39 + 8LL * (unsigned int)v35), 0, 8LL * (_QWORD)v43);
    result = *((unsigned int *)a1 + 20);
    if ( (unsigned int)v6 <= (unsigned int)result )
      goto LABEL_38;
  }
  else
  {
LABEL_37:
    result = *((unsigned int *)a1 + 20);
    if ( (unsigned int)v6 <= (unsigned int)result )
      goto LABEL_38;
  }
  v44 = (unsigned __int64)a1[9];
  v45 = (unsigned int)(result + 63) >> 6;
  if ( v44 > v45 )
  {
    v48 = v44 - v45;
    if ( v48 )
    {
      memset(&a1[8][v45], 0, 8 * v48);
      result = *((unsigned int *)a1 + 20);
    }
  }
  if ( (result & 0x3F) != 0 )
  {
    a1[8][v45 - 1] &= ~(-1LL << (result & 0x3F));
    result = *((unsigned int *)a1 + 20);
    *((_DWORD *)a1 + 20) = v6;
    if ( (unsigned int)v6 >= (unsigned int)result )
      goto LABEL_39;
    goto LABEL_60;
  }
LABEL_38:
  *((_DWORD *)a1 + 20) = v6;
  if ( (unsigned int)v6 >= (unsigned int)result )
    goto LABEL_39;
LABEL_60:
  result = (unsigned __int64)a1[9];
  v46 = (unsigned int)(v6 + 63) >> 6;
  if ( result > v46 )
  {
    result -= v46;
    if ( result )
    {
      result = (unsigned __int64)memset(&a1[8][v46], 0, 8 * result);
      v7 = *((_DWORD *)a1 + 20);
    }
  }
  v47 = v7 & 0x3F;
  if ( v47 )
  {
    result = ~(-1LL << v47);
    a1[8][v46 - 1] &= result;
  }
LABEL_39:
  if ( src )
    return j_j___libc_free_0(src, v56 - (_BYTE *)src);
  return result;
}

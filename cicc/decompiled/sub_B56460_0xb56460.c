// Function: sub_B56460
// Address: 0xb56460
//
__int64 __fastcall sub_B56460(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rcx
  __int64 v5; // r15
  size_t v6; // r12
  _QWORD *v7; // rax
  _BYTE *v8; // rdi
  __int64 v9; // rdx
  size_t v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rax
  char *v13; // r12
  char *v14; // r13
  __int64 result; // rax
  char *v16; // r14
  __int64 v17; // rcx
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rdi
  size_t v21; // rdx
  char *v22; // r8
  size_t v23; // rdx
  unsigned __int64 v24; // rsi
  bool v25; // cf
  unsigned __int64 v26; // rax
  size_t v27; // r10
  __int64 v28; // r15
  __int64 v29; // rax
  char *v30; // r11
  char *v31; // r9
  __int64 v32; // r15
  char *v33; // rax
  char *v34; // rdx
  char *v35; // rsi
  char *v36; // rax
  char *v37; // rdi
  char *v38; // r12
  size_t v39; // [rsp+8h] [rbp-88h]
  __int64 v40; // [rsp+10h] [rbp-80h]
  char *v41; // [rsp+10h] [rbp-80h]
  char *v42; // [rsp+18h] [rbp-78h]
  __int64 v43; // [rsp+18h] [rbp-78h]
  size_t v44; // [rsp+20h] [rbp-70h]
  char *v45; // [rsp+20h] [rbp-70h]
  _BYTE *v46; // [rsp+28h] [rbp-68h]
  char *v47; // [rsp+28h] [rbp-68h]
  size_t v48; // [rsp+28h] [rbp-68h]
  char *v49; // [rsp+28h] [rbp-68h]
  size_t v50; // [rsp+38h] [rbp-58h] BYREF
  _QWORD *v51; // [rsp+40h] [rbp-50h] BYREF
  size_t n; // [rsp+48h] [rbp-48h]
  _QWORD src[8]; // [rsp+50h] [rbp-40h] BYREF

  v2 = (_BYTE *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  v5 = *(_QWORD *)(a2 + 16);
  *(_BYTE *)(a1 + 16) = 0;
  v6 = *(_QWORD *)v5;
  v51 = src;
  v50 = v6;
  if ( v6 > 0xF )
  {
    v19 = sub_22409D0(&v51, &v50, 0);
    v2 = (_BYTE *)(a1 + 16);
    v51 = (_QWORD *)v19;
    v20 = (_QWORD *)v19;
    src[0] = v50;
  }
  else
  {
    if ( v6 == 1 )
    {
      LOBYTE(src[0]) = *(_BYTE *)(v5 + 16);
      v7 = src;
      goto LABEL_4;
    }
    if ( !v6 )
    {
      v7 = src;
      goto LABEL_4;
    }
    v20 = src;
  }
  v46 = v2;
  memcpy(v20, (const void *)(v5 + 16), v6);
  v6 = v50;
  v7 = v51;
  v2 = v46;
LABEL_4:
  n = v6;
  *((_BYTE *)v7 + v6) = 0;
  v8 = *(_BYTE **)a1;
  if ( v51 == src )
  {
    v21 = n;
    if ( n )
    {
      if ( n == 1 )
        *v8 = src[0];
      else
        memcpy(v8, src, n);
      v21 = n;
      v8 = *(_BYTE **)a1;
    }
    *(_QWORD *)(a1 + 8) = v21;
    v8[v21] = 0;
    v8 = v51;
  }
  else
  {
    v9 = src[0];
    v10 = n;
    if ( v2 == v8 )
    {
      *(_QWORD *)a1 = v51;
      *(_QWORD *)(a1 + 8) = v10;
      *(_QWORD *)(a1 + 16) = v9;
    }
    else
    {
      v11 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)a1 = v51;
      *(_QWORD *)(a1 + 8) = v10;
      *(_QWORD *)(a1 + 16) = v9;
      if ( v8 )
      {
        v51 = v8;
        src[0] = v11;
        goto LABEL_8;
      }
    }
    v51 = src;
    v8 = src;
  }
LABEL_8:
  n = 0;
  *v8 = 0;
  if ( v51 != src )
    j_j___libc_free_0(v51, src[0] + 1LL);
  v12 = *(_QWORD *)(a2 + 8);
  v13 = *(char **)a2;
  v14 = *(char **)(a1 + 40);
  result = 32 * v12;
  v16 = &v13[result];
  if ( v13 != &v13[result] )
  {
    v17 = *(_QWORD *)(a1 + 48);
    v18 = result >> 5;
    if ( v18 <= (v17 - (__int64)v14) >> 3 )
    {
      do
      {
        if ( v14 )
          *(_QWORD *)v14 = *(_QWORD *)v13;
        v13 += 32;
        v14 += 8;
      }
      while ( v16 != v13 );
      result = 8 * v18;
      *(_QWORD *)(a1 + 40) += result;
      return result;
    }
    v22 = *(char **)(a1 + 32);
    v23 = v14 - v22;
    v24 = (v14 - v22) >> 3;
    if ( v18 > 0xFFFFFFFFFFFFFFFLL - v24 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v18 < v24 )
      v18 = (v14 - v22) >> 3;
    v25 = __CFADD__(v24, v18);
    v26 = v24 + v18;
    v27 = v26;
    if ( v25 )
    {
      v28 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v26 )
      {
        v30 = *(char **)(a1 + 40);
        v32 = 0;
        v31 = 0;
        goto LABEL_35;
      }
      if ( v26 > 0xFFFFFFFFFFFFFFFLL )
        v26 = 0xFFFFFFFFFFFFFFFLL;
      v28 = 8 * v26;
    }
    v29 = sub_22077B0(v28);
    v30 = *(char **)(a1 + 40);
    v22 = *(char **)(a1 + 32);
    v17 = *(_QWORD *)(a1 + 48);
    v31 = (char *)v29;
    v32 = v29 + v28;
    v23 = v14 - v22;
    v27 = v30 - v14;
LABEL_35:
    if ( v14 != v22 )
    {
      v39 = v27;
      v40 = v17;
      v42 = v30;
      v44 = v23;
      v47 = v22;
      v33 = (char *)memmove(v31, v22, v23);
      v27 = v39;
      v17 = v40;
      v30 = v42;
      v23 = v44;
      v31 = v33;
      v22 = v47;
    }
    v34 = &v31[v23];
    v35 = v13;
    v36 = v34;
    do
    {
      if ( v36 )
        *(_QWORD *)v36 = *(_QWORD *)v13;
      v13 += 32;
      v36 += 8;
    }
    while ( v16 != v13 );
    result = (unsigned __int64)(v16 - v35 - 32) >> 5;
    v37 = &v34[8 * result + 8];
    if ( v14 != v30 )
    {
      v41 = v31;
      v43 = v17;
      v45 = v22;
      v48 = v27;
      result = (__int64)memcpy(v37, v14, v27);
      v31 = v41;
      v17 = v43;
      v22 = v45;
      v27 = v48;
      v37 = (char *)result;
    }
    v38 = &v37[v27];
    if ( v22 )
    {
      v49 = v31;
      result = j_j___libc_free_0(v22, v17 - (_QWORD)v22);
      v31 = v49;
    }
    *(_QWORD *)(a1 + 40) = v38;
    *(_QWORD *)(a1 + 48) = v32;
    *(_QWORD *)(a1 + 32) = v31;
  }
  return result;
}

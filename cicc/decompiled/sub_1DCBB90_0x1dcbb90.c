// Function: sub_1DCBB90
// Address: 0x1dcbb90
//
unsigned __int64 __fastcall sub_1DCBB90(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  _BYTE *v9; // rsi
  unsigned int v10; // r15d
  unsigned __int64 result; // rax
  __int64 v12; // rdx
  __int64 *v13; // rdi
  __int64 *v14; // rdx
  unsigned int v15; // esi
  unsigned int v16; // ecx
  __int64 v17; // r15
  _QWORD *v18; // r12
  __int64 v19; // rcx
  unsigned __int64 v20; // r10
  __int64 v21; // rbx
  _QWORD *v22; // r8
  signed __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rsi
  bool v26; // cf
  unsigned __int64 v27; // rax
  __int64 v28; // r14
  __int64 v29; // rax
  char *v30; // r9
  __int64 v31; // r14
  char *v32; // rax
  char *v33; // rdi
  __int64 v34; // r10
  char *v35; // rax
  unsigned __int64 v36; // rbx
  char *v37; // rbx
  unsigned __int64 v38; // [rsp+0h] [rbp-50h]
  __int64 v39; // [rsp+8h] [rbp-48h]
  __int64 v40; // [rsp+10h] [rbp-40h]
  signed __int64 v41; // [rsp+10h] [rbp-40h]
  char *v42; // [rsp+10h] [rbp-40h]
  __int64 v43; // [rsp+18h] [rbp-38h]
  _QWORD *v44; // [rsp+18h] [rbp-38h]
  _QWORD *v45; // [rsp+18h] [rbp-38h]
  char *v46; // [rsp+18h] [rbp-38h]

  v9 = (_BYTE *)a2[4];
  v10 = *(_DWORD *)(a4 + 48);
  result = (a2[5] - (__int64)v9) >> 3;
  if ( (_DWORD)result )
  {
    v12 = (__int64)&v9[8 * (unsigned int)(result - 1) + 8];
    while ( 1 )
    {
      result = *(_QWORD *)v9;
      if ( a4 == *(_QWORD *)(*(_QWORD *)v9 + 24LL) )
        break;
      v9 += 8;
      if ( (_BYTE *)v12 == v9 )
        goto LABEL_6;
    }
    result = (unsigned __int64)sub_1DCBB50((__int64)(a2 + 4), v9);
  }
LABEL_6:
  if ( a4 != a3 )
  {
    v13 = (__int64 *)a2[1];
    v14 = a2 + 1;
    if ( v13 == a2 + 1 )
      goto LABEL_22;
    result = *a2;
    if ( v14 == (__int64 *)*a2 )
    {
      result = a2[2];
      *a2 = result;
    }
    v15 = *(_DWORD *)(result + 16);
    v16 = v10 >> 7;
    if ( v10 >> 7 == v15 )
    {
      if ( v14 == (__int64 *)result )
        goto LABEL_22;
      goto LABEL_19;
    }
    if ( v16 < v15 )
    {
      if ( v13 == (__int64 *)result )
      {
        *a2 = result;
        goto LABEL_18;
      }
      do
        result = *(_QWORD *)(result + 8);
      while ( v13 != (__int64 *)result && v16 < *(_DWORD *)(result + 16) );
    }
    else
    {
      if ( v14 == (__int64 *)result )
      {
LABEL_21:
        *a2 = result;
        goto LABEL_22;
      }
      while ( v15 < v16 )
      {
        result = *(_QWORD *)result;
        if ( v14 == (__int64 *)result )
          goto LABEL_21;
        v15 = *(_DWORD *)(result + 16);
      }
    }
    *a2 = result;
    if ( v14 == (__int64 *)result )
      goto LABEL_22;
LABEL_18:
    if ( v16 != *(_DWORD *)(result + 16) )
      goto LABEL_22;
LABEL_19:
    if ( (*(_QWORD *)(result + 8LL * ((v10 >> 6) & 1) + 24) & (1LL << v10)) != 0 )
      return result;
LABEL_22:
    sub_1369D60(a2, v10);
    result = *(_QWORD *)(a4 + 64);
    v17 = *(_QWORD *)(a4 + 72);
    v18 = *(_QWORD **)(a5 + 8);
    if ( v17 == result )
      return result;
    v19 = v17 - result;
    v20 = v19 >> 3;
    result = *(_QWORD *)(a5 + 16) - (_QWORD)v18;
    v21 = v19 >> 3;
    if ( result >= v19 )
    {
      if ( v19 > 0 )
      {
        do
        {
          result = *(_QWORD *)(v17 - 8 * v20 + 8 * v21 - 8);
          *v18++ = result;
          --v21;
        }
        while ( v21 );
        v18 = *(_QWORD **)(a5 + 8);
      }
      *(_QWORD *)(a5 + 8) = (char *)v18 + v19;
      return result;
    }
    v22 = *(_QWORD **)a5;
    v23 = (signed __int64)v18 - *(_QWORD *)a5;
    v24 = v23 >> 3;
    if ( v20 > 0xFFFFFFFFFFFFFFFLL - (v23 >> 3) )
      sub_4262D8((__int64)"vector::_M_range_insert");
    v25 = ((__int64)v18 - *(_QWORD *)a5) >> 3;
    if ( v20 >= v24 )
      v25 = v19 >> 3;
    v26 = __CFADD__(v25, v24);
    v27 = v25 + v24;
    if ( v26 )
    {
      v28 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v27 )
      {
        v31 = 0;
        v30 = 0;
        goto LABEL_41;
      }
      if ( v27 > 0xFFFFFFFFFFFFFFFLL )
        v27 = 0xFFFFFFFFFFFFFFFLL;
      v28 = 8 * v27;
    }
    v40 = v19 >> 3;
    v43 = v19;
    v29 = sub_22077B0(v28);
    v22 = *(_QWORD **)a5;
    v19 = v43;
    v20 = v40;
    v30 = (char *)v29;
    v31 = v29 + v28;
    v23 = (signed __int64)v18 - *(_QWORD *)a5;
LABEL_41:
    if ( v18 != v22 )
    {
      v38 = v20;
      v39 = v19;
      v41 = v23;
      v44 = v22;
      v32 = (char *)memmove(v30, v22, v23);
      v20 = v38;
      v19 = v39;
      v23 = v41;
      v22 = v44;
      v30 = v32;
    }
    v33 = &v30[v23];
    if ( v19 > 0 )
    {
      v34 = -(__int64)v20;
      v35 = &v30[v23];
      do
      {
        v35 += 8;
        *((_QWORD *)v35 - 1) = *(_QWORD *)(v17 + 8 * v34 + 8 * v21-- - 8);
      }
      while ( v21 );
      v33 += v19;
    }
    result = *(_QWORD *)(a5 + 8);
    v36 = result - (_QWORD)v18;
    if ( v18 != (_QWORD *)result )
    {
      v42 = v30;
      v45 = v22;
      result = (unsigned __int64)memmove(v33, v18, *(_QWORD *)(a5 + 8) - (_QWORD)v18);
      v30 = v42;
      v22 = v45;
      v33 = (char *)result;
    }
    v37 = &v33[v36];
    if ( v22 )
    {
      v46 = v30;
      result = j_j___libc_free_0(v22, *(_QWORD *)(a5 + 16) - (_QWORD)v22);
      v30 = v46;
    }
    *(_QWORD *)a5 = v30;
    *(_QWORD *)(a5 + 8) = v37;
    *(_QWORD *)(a5 + 16) = v31;
  }
  return result;
}

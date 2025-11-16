// Function: sub_22B5D50
// Address: 0x22b5d50
//
__int64 __fastcall sub_22B5D50(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  bool v9; // zf
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  _BYTE *v15; // rsi
  __int64 v16; // rax
  char v17; // dl
  unsigned int v18; // eax
  unsigned int *v19; // rdx
  unsigned int v20; // r14d
  unsigned int v22; // esi
  int v23; // eax
  __int64 *v24; // rdx
  int v25; // eax
  const void *v26; // r15
  signed __int64 v27; // r12
  __int64 v28; // rcx
  __int64 v29; // rax
  bool v30; // cf
  unsigned __int64 v31; // rax
  char *v32; // rbx
  char *v33; // rcx
  __int64 v34; // r8
  char *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  unsigned __int64 v39; // rbx
  char *v40; // [rsp+0h] [rbp-70h]
  __int64 v41; // [rsp+8h] [rbp-68h]
  __int64 v42; // [rsp+8h] [rbp-68h]
  __int64 v43; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v44; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v45; // [rsp+28h] [rbp-48h] BYREF
  __int64 v46; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v47; // [rsp+38h] [rbp-38h]

  v9 = *(_BYTE *)(a1 + 73) == 0;
  *(_BYTE *)(a1 + 72) = 0;
  if ( !v9 )
    *(_BYTE *)(a1 + 74) = 1;
  *(_BYTE *)(a1 + 73) = 1;
  v10 = *a2;
  if ( *a2 )
    v10 = *a2 - 24;
  v11 = sub_22B4800(a1, v10, 1, *(_QWORD *)(a1 + 96), a5, a6);
  v15 = *(_BYTE **)(a4 + 8);
  v43 = v11;
  if ( v15 == *(_BYTE **)(a4 + 16) )
  {
    sub_22B18C0(a4, v15, &v43);
  }
  else
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = v11;
      v15 = *(_BYTE **)(a4 + 8);
    }
    *(_QWORD *)(a4 + 8) = v15 + 8;
  }
  v16 = *a2;
  if ( !*a2 )
    goto LABEL_40;
  v17 = *(_BYTE *)(v16 - 24);
  if ( v17 != 31 )
  {
    if ( v17 != 85 )
      goto LABEL_12;
    goto LABEL_37;
  }
  sub_22AE820(v43, a1 + 40);
  v16 = *a2;
  if ( !*a2 )
LABEL_40:
    BUG();
  if ( *(_BYTE *)(v16 - 24) != 85 )
  {
LABEL_12:
    if ( *(_BYTE *)(v16 - 24) != 84 )
      goto LABEL_13;
    goto LABEL_39;
  }
LABEL_37:
  sub_22AE9F0(v43, *(_BYTE *)(a1 + 75));
  if ( !*a2 )
    BUG();
  if ( *(_BYTE *)(*a2 - 24) == 84 )
LABEL_39:
    sub_22AEFC0(v43, a1 + 40);
LABEL_13:
  v18 = *(_DWORD *)(a1 + 4);
  v46 = v43;
  v47 = v18;
  if ( !(unsigned __int8)sub_22B30A0(a1 + 8, &v46, &v44, v12, v13, v14) )
  {
    v22 = *(_DWORD *)(a1 + 32);
    v23 = *(_DWORD *)(a1 + 24);
    v24 = v44;
    ++*(_QWORD *)(a1 + 8);
    v25 = v23 + 1;
    v45 = v24;
    if ( 4 * v25 >= 3 * v22 )
    {
      v22 *= 2;
    }
    else if ( v22 - *(_DWORD *)(a1 + 28) - v25 > v22 >> 3 )
    {
      goto LABEL_21;
    }
    sub_22B5BC0(a1 + 8, v22);
    sub_22B30A0(a1 + 8, &v46, &v45, v36, v37, v38);
    v24 = v45;
    v25 = *(_DWORD *)(a1 + 24) + 1;
LABEL_21:
    *(_DWORD *)(a1 + 24) = v25;
    if ( *v24 )
      --*(_DWORD *)(a1 + 28);
    *v24 = v46;
    v20 = v47;
    *((_DWORD *)v24 + 2) = v47;
    ++*(_DWORD *)(a1 + 4);
    v19 = *(unsigned int **)(a3 + 8);
    if ( v19 != *(unsigned int **)(a3 + 16) )
      goto LABEL_15;
LABEL_24:
    v26 = *(const void **)a3;
    v27 = (signed __int64)v19 - *(_QWORD *)a3;
    v28 = v27 >> 2;
    if ( v27 >> 2 == 0x1FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v29 = 1;
    if ( v28 )
      v29 = v27 >> 2;
    v30 = __CFADD__(v28, v29);
    v31 = v28 + v29;
    if ( v30 )
    {
      v39 = 0x7FFFFFFFFFFFFFFCLL;
    }
    else
    {
      if ( !v31 )
      {
        v32 = 0;
        v33 = 0;
        goto LABEL_30;
      }
      if ( v31 > 0x1FFFFFFFFFFFFFFFLL )
        v31 = 0x1FFFFFFFFFFFFFFFLL;
      v39 = 4 * v31;
    }
    v33 = (char *)sub_22077B0(v39);
    v32 = &v33[v39];
LABEL_30:
    if ( &v33[v27] )
      *(_DWORD *)&v33[v27] = v20;
    v34 = (__int64)&v33[v27 + 4];
    if ( v27 > 0 )
    {
      v41 = (__int64)&v33[v27 + 4];
      v35 = (char *)memmove(v33, v26, v27);
      v34 = v41;
      v33 = v35;
    }
    else if ( !v26 )
    {
LABEL_34:
      *(_QWORD *)a3 = v33;
      *(_QWORD *)(a3 + 8) = v34;
      *(_QWORD *)(a3 + 16) = v32;
      return v20;
    }
    v40 = v33;
    v42 = v34;
    j_j___libc_free_0((unsigned __int64)v26);
    v33 = v40;
    v34 = v42;
    goto LABEL_34;
  }
  v19 = *(unsigned int **)(a3 + 8);
  v20 = *((_DWORD *)v44 + 2);
  if ( v19 == *(unsigned int **)(a3 + 16) )
    goto LABEL_24;
LABEL_15:
  if ( v19 )
  {
    *v19 = v20;
    v19 = *(unsigned int **)(a3 + 8);
  }
  *(_QWORD *)(a3 + 8) = v19 + 1;
  return v20;
}

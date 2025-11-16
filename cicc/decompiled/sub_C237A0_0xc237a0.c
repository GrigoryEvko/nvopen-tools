// Function: sub_C237A0
// Address: 0xc237a0
//
__int64 __fastcall sub_C237A0(__int64 *a1, char *a2, char **a3)
{
  char *v5; // r15
  char *v6; // r13
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  char *v10; // r12
  bool v11; // cf
  unsigned __int64 v12; // rax
  signed __int64 v13; // rdx
  __int64 v14; // rbx
  _QWORD *v15; // rdx
  __int64 v16; // rbx
  char *v17; // r8
  __int64 v18; // rax
  void *v19; // rdi
  unsigned int v20; // r11d
  size_t v21; // rdx
  unsigned int v22; // r8d
  void *v23; // rdi
  size_t v24; // rdx
  char *i; // r12
  __int64 v27; // rbx
  __int64 v28; // rax
  char *v29; // [rsp+8h] [rbp-58h]
  char *v30; // [rsp+8h] [rbp-58h]
  unsigned int v31; // [rsp+10h] [rbp-50h]
  unsigned int v32; // [rsp+10h] [rbp-50h]
  char *v33; // [rsp+10h] [rbp-50h]
  char *v34; // [rsp+18h] [rbp-48h]
  unsigned int v35; // [rsp+18h] [rbp-48h]
  unsigned int v36; // [rsp+18h] [rbp-48h]
  char *v37; // [rsp+18h] [rbp-48h]
  signed __int64 v38; // [rsp+18h] [rbp-48h]
  char *v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+28h] [rbp-38h]

  v5 = (char *)a1[1];
  v6 = (char *)*a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)&v5[-*a1] >> 3);
  if ( v7 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v10 = a2;
  if ( v7 )
    v8 = 0xCCCCCCCCCCCCCCCDLL * ((a1[1] - *a1) >> 3);
  v11 = __CFADD__(v8, v7);
  v12 = v8 - 0x3333333333333333LL * ((a1[1] - *a1) >> 3);
  v13 = a2 - v6;
  if ( v11 )
  {
    v27 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_41:
    v33 = a2;
    v38 = a2 - v6;
    v28 = sub_22077B0(v27);
    v13 = v38;
    a2 = v33;
    v41 = v28;
    v40 = v28 + v27;
    v14 = v28 + 40;
    goto LABEL_7;
  }
  if ( v12 )
  {
    if ( v12 > 0x333333333333333LL )
      v12 = 0x333333333333333LL;
    v27 = 40 * v12;
    goto LABEL_41;
  }
  v40 = 0;
  v14 = 40;
  v41 = 0;
LABEL_7:
  v15 = (_QWORD *)(v41 + v13);
  if ( v15 )
  {
    *v15 = v15 + 2;
    v15[1] = 0x100000000LL;
    if ( *((_DWORD *)a3 + 2) )
    {
      v39 = a2;
      sub_C1E9B0((__int64)v15, a3);
      a2 = v39;
    }
  }
  if ( a2 != v6 )
  {
    v16 = v41;
    v17 = v6;
    while ( 1 )
    {
      if ( v16
        && (v19 = (void *)(v16 + 16),
            *(_DWORD *)(v16 + 8) = 0,
            *(_QWORD *)v16 = v16 + 16,
            *(_DWORD *)(v16 + 12) = 1,
            (v20 = *((_DWORD *)v17 + 2)) != 0)
        && v17 != (char *)v16 )
      {
        v21 = 24;
        if ( v20 == 1 )
          goto LABEL_18;
        a3 = (char **)(v16 + 16);
        v30 = a2;
        v32 = *((_DWORD *)v17 + 2);
        v37 = v17;
        sub_C8D5F0(v16, v16 + 16, v20, 24);
        v17 = v37;
        v19 = *(void **)v16;
        v20 = v32;
        a2 = v30;
        v21 = 24LL * *((unsigned int *)v37 + 2);
        if ( v21 )
        {
LABEL_18:
          a3 = *(char ***)v17;
          v29 = a2;
          v31 = v20;
          v34 = v17;
          memcpy(v19, *(const void **)v17, v21);
          a2 = v29;
          v17 = v34;
          *(_DWORD *)(v16 + 8) = v31;
        }
        else
        {
          *(_DWORD *)(v16 + 8) = v32;
        }
        v17 += 40;
        v18 = v16 + 40;
        if ( a2 == v17 )
        {
LABEL_20:
          v14 = v16 + 80;
          break;
        }
      }
      else
      {
        v17 += 40;
        v18 = v16 + 40;
        if ( a2 == v17 )
          goto LABEL_20;
      }
      v16 = v18;
    }
  }
  if ( a2 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v22 = *((_DWORD *)v10 + 2);
        v23 = (void *)(v14 + 16);
        *(_DWORD *)(v14 + 8) = 0;
        *(_QWORD *)v14 = v14 + 16;
        *(_DWORD *)(v14 + 12) = 1;
        if ( v22 )
        {
          if ( v10 != (char *)v14 )
            break;
        }
        v10 += 40;
        v14 += 40;
        if ( v5 == v10 )
          goto LABEL_29;
      }
      v24 = 24;
      if ( v22 == 1
        || (a3 = (char **)(v14 + 16),
            v36 = v22,
            sub_C8D5F0(v14, v14 + 16, v22, 24),
            v23 = *(void **)v14,
            v22 = v36,
            (v24 = 24LL * *((unsigned int *)v10 + 2)) != 0) )
      {
        a3 = *(char ***)v10;
        v35 = v22;
        memcpy(v23, *(const void **)v10, v24);
        *(_DWORD *)(v14 + 8) = v35;
      }
      else
      {
        *(_DWORD *)(v14 + 8) = v36;
      }
      v10 += 40;
      v14 += 40;
    }
    while ( v5 != v10 );
  }
LABEL_29:
  for ( i = v6; v5 != i; i += 40 )
  {
    if ( *(char **)i != i + 16 )
      _libc_free(*(_QWORD *)i, a3);
  }
  if ( v6 )
    j_j___libc_free_0(v6, a1[2] - (_QWORD)v6);
  a1[1] = v14;
  *a1 = v41;
  a1[2] = v40;
  return v40;
}

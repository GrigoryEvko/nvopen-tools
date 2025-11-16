// Function: sub_1740580
// Address: 0x1740580
//
__int64 __fastcall sub_1740580(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  __int64 v4; // rbx
  _BYTE *v5; // rcx
  __int64 v6; // r8
  size_t v7; // r14
  _QWORD *v8; // rax
  _BYTE *v9; // rdi
  __int64 v10; // rdx
  char *v11; // r14
  char *v12; // r13
  __int64 v13; // rax
  char *v14; // rcx
  unsigned __int64 v15; // rax
  __int64 result; // rax
  __int64 v17; // rax
  _QWORD *v18; // rdi
  size_t v19; // rdx
  char *v20; // r8
  size_t v21; // rdx
  unsigned __int64 v22; // rsi
  bool v23; // cf
  unsigned __int64 v24; // rax
  __int64 v25; // r15
  __int64 v26; // rax
  char *v27; // r9
  __int64 v28; // r15
  char *v29; // rax
  char *v30; // rdi
  char *v31; // rax
  char *v32; // rdx
  char *v33; // rcx
  __int64 v34; // r14
  char *v35; // rax
  char *v36; // r14
  char *v37; // [rsp+8h] [rbp-78h]
  __int64 v38; // [rsp+10h] [rbp-70h]
  size_t v39; // [rsp+10h] [rbp-70h]
  char *v40; // [rsp+10h] [rbp-70h]
  _BYTE *v41; // [rsp+18h] [rbp-68h]
  char *v42; // [rsp+18h] [rbp-68h]
  char *v43; // [rsp+18h] [rbp-68h]
  char *v44; // [rsp+18h] [rbp-68h]
  char *v45; // [rsp+18h] [rbp-68h]
  size_t v46; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v47; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  _QWORD src[8]; // [rsp+40h] [rbp-40h] BYREF

  v3 = *(_DWORD *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= v3 )
  {
    sub_1740340(a1, 0);
    v3 = *(_DWORD *)(a1 + 8);
  }
  v4 = *(_QWORD *)a1 + 56LL * v3;
  if ( v4 )
  {
    v5 = (_BYTE *)(v4 + 16);
    *(_QWORD *)(v4 + 8) = 0;
    *(_QWORD *)v4 = v4 + 16;
    *(_QWORD *)(v4 + 32) = 0;
    *(_QWORD *)(v4 + 40) = 0;
    *(_QWORD *)(v4 + 48) = 0;
    v6 = *(_QWORD *)(a2 + 16);
    *(_BYTE *)(v4 + 16) = 0;
    v7 = *(_QWORD *)v6;
    v47 = src;
    v46 = v7;
    if ( v7 > 0xF )
    {
      v38 = v6;
      v17 = sub_22409D0(&v47, &v46, 0);
      v5 = (_BYTE *)(v4 + 16);
      v6 = v38;
      v47 = (_QWORD *)v17;
      v18 = (_QWORD *)v17;
      src[0] = v46;
    }
    else
    {
      if ( v7 == 1 )
      {
        LOBYTE(src[0]) = *(_BYTE *)(v6 + 16);
        v8 = src;
        goto LABEL_7;
      }
      if ( !v7 )
      {
        v8 = src;
LABEL_7:
        n = v7;
        *((_BYTE *)v8 + v7) = 0;
        v9 = *(_BYTE **)v4;
        if ( v47 == src )
        {
          v19 = n;
          if ( n )
          {
            if ( n == 1 )
              *v9 = src[0];
            else
              memcpy(v9, src, n);
            v19 = n;
            v9 = *(_BYTE **)v4;
          }
          *(_QWORD *)(v4 + 8) = v19;
          v9[v19] = 0;
          v9 = v47;
        }
        else
        {
          if ( v5 == v9 )
          {
            *(_QWORD *)v4 = v47;
            *(_QWORD *)(v4 + 8) = n;
            *(_QWORD *)(v4 + 16) = src[0];
          }
          else
          {
            *(_QWORD *)v4 = v47;
            v10 = *(_QWORD *)(v4 + 16);
            *(_QWORD *)(v4 + 8) = n;
            *(_QWORD *)(v4 + 16) = src[0];
            if ( v9 )
            {
              v47 = v9;
              src[0] = v10;
              goto LABEL_11;
            }
          }
          v47 = src;
          v9 = src;
        }
LABEL_11:
        n = 0;
        *v9 = 0;
        if ( v47 != src )
          j_j___libc_free_0(v47, src[0] + 1LL);
        v11 = *(char **)a2;
        v12 = *(char **)(v4 + 40);
        v13 = 24LL * *(_QWORD *)(a2 + 8);
        v14 = (char *)(*(_QWORD *)a2 + v13);
        if ( v11 == &v11[v13] )
          goto LABEL_19;
        v15 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
        if ( v15 <= (__int64)(*(_QWORD *)(v4 + 48) - (_QWORD)v12) >> 3 )
        {
          do
          {
            if ( v12 )
              *(_QWORD *)v12 = *(_QWORD *)v11;
            v11 += 24;
            v12 += 8;
          }
          while ( v14 != v11 );
          *(_QWORD *)(v4 + 40) += 8 * v15;
LABEL_19:
          v3 = *(_DWORD *)(a1 + 8);
          goto LABEL_20;
        }
        v20 = *(char **)(v4 + 32);
        v21 = v12 - v20;
        v22 = (v12 - v20) >> 3;
        if ( v15 > 0xFFFFFFFFFFFFFFFLL - v22 )
          sub_4262D8((__int64)"vector::_M_range_insert");
        if ( v15 < v22 )
          v15 = (v12 - v20) >> 3;
        v23 = __CFADD__(v22, v15);
        v24 = v22 + v15;
        if ( v23 )
        {
          v25 = 0x7FFFFFFFFFFFFFF8LL;
        }
        else
        {
          if ( !v24 )
          {
            v28 = 0;
            v27 = 0;
            goto LABEL_37;
          }
          if ( v24 > 0xFFFFFFFFFFFFFFFLL )
            v24 = 0xFFFFFFFFFFFFFFFLL;
          v25 = 8 * v24;
        }
        v42 = v14;
        v26 = sub_22077B0(v25);
        v20 = *(char **)(v4 + 32);
        v14 = v42;
        v27 = (char *)v26;
        v28 = v26 + v25;
        v21 = v12 - v20;
LABEL_37:
        if ( v12 != v20 )
        {
          v37 = v14;
          v39 = v21;
          v43 = v20;
          v29 = (char *)memmove(v27, v20, v21);
          v14 = v37;
          v21 = v39;
          v20 = v43;
          v27 = v29;
        }
        v30 = &v27[v21];
        v31 = v11;
        v32 = &v27[v21];
        do
        {
          if ( v32 )
            *(_QWORD *)v32 = *(_QWORD *)v31;
          v31 += 24;
          v32 += 8;
        }
        while ( v14 != v31 );
        v33 = &v30[0x5555555555555558LL * ((unsigned __int64)(v14 - v11 - 24) >> 3) + 8];
        v34 = *(_QWORD *)(v4 + 40) - (_QWORD)v12;
        if ( v12 != *(char **)(v4 + 40) )
        {
          v40 = v27;
          v44 = v20;
          v35 = (char *)memcpy(v33, v12, *(_QWORD *)(v4 + 40) - (_QWORD)v12);
          v27 = v40;
          v20 = v44;
          v33 = v35;
        }
        v36 = &v33[v34];
        if ( v20 )
        {
          v45 = v27;
          j_j___libc_free_0(v20, *(_QWORD *)(v4 + 48) - (_QWORD)v20);
          v27 = v45;
        }
        *(_QWORD *)(v4 + 32) = v27;
        *(_QWORD *)(v4 + 40) = v36;
        *(_QWORD *)(v4 + 48) = v28;
        goto LABEL_19;
      }
      v18 = src;
    }
    v41 = v5;
    memcpy(v18, (const void *)(v6 + 16), v7);
    v7 = v46;
    v8 = v47;
    v5 = v41;
    goto LABEL_7;
  }
LABEL_20:
  result = v3 + 1;
  *(_DWORD *)(a1 + 8) = result;
  return result;
}

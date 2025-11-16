// Function: sub_104D880
// Address: 0x104d880
//
__int64 __fastcall sub_104D880(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rdx
  int v4; // r8d
  __int64 v5; // r14
  void *v6; // rdx
  char *v7; // r15
  __int64 v8; // rsi
  char *v9; // rbx
  char *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rcx
  size_t v16; // rdx
  unsigned __int8 *v17; // rsi
  __int64 v18; // rax
  _WORD *v19; // rdx
  __int64 v20; // rdi
  __int64 result; // rax
  __int64 *v22; // rax
  __int64 *v23; // r12
  __int64 *v24; // rbx
  const char *v25; // r8
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r9
  unsigned __int64 v29; // rdx
  const char **v30; // rax
  const char *v31; // [rsp+0h] [rbp-190h]
  __int64 v32; // [rsp+8h] [rbp-188h]
  unsigned __int8 *v34; // [rsp+30h] [rbp-160h] BYREF
  size_t v35; // [rsp+38h] [rbp-158h]
  _QWORD v36[2]; // [rsp+40h] [rbp-150h] BYREF
  void *base; // [rsp+50h] [rbp-140h] BYREF
  size_t nmemb; // [rsp+58h] [rbp-138h]
  _BYTE v39[296]; // [rsp+60h] [rbp-130h] BYREF

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_DWORD *)(v3 + 648);
  base = v39;
  nmemb = 0x1000000000LL;
  if ( v4 )
  {
    v22 = *(__int64 **)(v3 + 640);
    v23 = &v22[2 * *(unsigned int *)(v3 + 656)];
    if ( v22 != v23 )
    {
      while ( 1 )
      {
        v24 = v22;
        if ( *v22 != -8192 && *v22 != -4096 )
          break;
        v22 += 2;
        if ( v23 == v22 )
          goto LABEL_2;
      }
      if ( v23 != v22 )
      {
        while ( 1 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 664) + 72LL * *((unsigned int *)v24 + 2)) + 8LL * (a2 >> 6))
              & (1LL << a2)) != 0 )
          {
            v25 = sub_BD5D20(*v24);
            v26 = (unsigned int)nmemb;
            v28 = v27;
            v29 = (unsigned int)nmemb + 1LL;
            if ( v29 > HIDWORD(nmemb) )
            {
              v31 = v25;
              v32 = v28;
              sub_C8D5F0((__int64)&base, v39, v29, 0x10u, (__int64)v25, v28);
              v26 = (unsigned int)nmemb;
              v25 = v31;
              v28 = v32;
            }
            v30 = (const char **)((char *)base + 16 * v26);
            *v30 = v25;
            v30[1] = (const char *)v28;
            LODWORD(nmemb) = nmemb + 1;
          }
          v24 += 2;
          if ( v24 == v23 )
            break;
          while ( *v24 == -4096 || *v24 == -8192 )
          {
            v24 += 2;
            if ( v23 == v24 )
              goto LABEL_31;
          }
          if ( v23 == v24 )
            break;
          v3 = *(_QWORD *)(a1 + 8);
        }
LABEL_31:
        if ( (unsigned int)nmemb > 1uLL )
          qsort(base, (unsigned int)nmemb, 0x10u, (__compar_fn_t)sub_A16990);
      }
    }
  }
LABEL_2:
  v5 = a3;
  v6 = *(void **)(a3 + 32);
  if ( *(_QWORD *)(a3 + 24) - (_QWORD)v6 <= 0xBu )
  {
    v5 = sub_CB6200(a3, (unsigned __int8 *)"  ; Alive: <", 0xCu);
  }
  else
  {
    qmemcpy(v6, "  ; Alive: <", 12);
    *(_QWORD *)(a3 + 32) += 12LL;
  }
  v7 = (char *)base;
  v35 = 0;
  v8 = 16LL * (unsigned int)nmemb;
  v34 = (unsigned __int8 *)v36;
  v9 = (char *)base + v8;
  LOBYTE(v36[0]) = 0;
  if ( base == (char *)base + v8 )
  {
    v17 = (unsigned __int8 *)v36;
    v16 = 0;
  }
  else
  {
    v10 = (char *)base;
    v11 = (v8 >> 4) - 1;
    do
    {
      v11 += *((_QWORD *)v10 + 1);
      v10 += 16;
    }
    while ( v9 != v10 );
    sub_2240E30(&v34, v11);
    v13 = *((_QWORD *)v7 + 1);
    v14 = *(_QWORD *)v7;
    if ( v13 > 0x3FFFFFFFFFFFFFFFLL - v35 )
LABEL_41:
      sub_4262D8((__int64)"basic_string::append");
    while ( 1 )
    {
      v7 += 16;
      sub_2241490(&v34, v14, v13, v12);
      if ( v9 == v7 )
        break;
      if ( v35 != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490(&v34, " ", 1, v15);
        v13 = *((_QWORD *)v7 + 1);
        v14 = *(_QWORD *)v7;
        if ( v13 <= 0x3FFFFFFFFFFFFFFFLL - v35 )
          continue;
      }
      goto LABEL_41;
    }
    v16 = v35;
    v17 = v34;
  }
  v18 = sub_CB6200(v5, v17, v16);
  v19 = *(_WORD **)(v18 + 32);
  v20 = v18;
  if ( *(_QWORD *)(v18 + 24) - (_QWORD)v19 <= 1u )
  {
    v17 = (unsigned __int8 *)">\n";
    result = sub_CB6200(v18, (unsigned __int8 *)">\n", 2u);
  }
  else
  {
    result = 2622;
    *v19 = 2622;
    *(_QWORD *)(v20 + 32) += 2LL;
  }
  if ( v34 != (unsigned __int8 *)v36 )
  {
    v17 = (unsigned __int8 *)(v36[0] + 1LL);
    result = j_j___libc_free_0(v34, v36[0] + 1LL);
  }
  if ( base != v39 )
    return _libc_free(base, v17);
  return result;
}

// Function: sub_1E15F80
// Address: 0x1e15f80
//
char *__fastcall sub_1E15F80(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  unsigned __int8 v5; // si
  __int64 **v6; // r13
  __int64 **v7; // r14
  __int64 **v8; // rcx
  __int64 *v9; // r15
  __int64 *v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int64 v14; // rax
  int v15; // r13d
  __int64 v16; // rax
  char *v17; // r14
  size_t v18; // rax
  size_t v19; // r15
  size_t v20; // rdx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rdx
  int v23; // eax
  __int64 **v24; // [rsp-50h] [rbp-50h]
  __int64 *v25; // [rsp-48h] [rbp-48h]
  __int64 v26; // [rsp-40h] [rbp-40h]
  int v27; // [rsp-40h] [rbp-40h]

  v2 = *(_BYTE *)(a1 + 49);
  if ( !v2 )
    return 0;
  v5 = *(_BYTE *)(a2 + 49);
  if ( !v5 )
    return 0;
  if ( v5 == (unsigned __int64)v2 )
  {
    v6 = *(__int64 ***)(a1 + 56);
    v7 = *(__int64 ***)(a2 + 56);
    v8 = &v6[v2];
    while ( 1 )
    {
      v9 = *v7;
      v10 = *v6;
      v11 = **v7;
      v12 = **v6;
      v13 = (v11 >> 2) & 1;
      if ( (v12 & 4) != 0 )
        break;
      v14 = v12 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_BYTE)v13 )
      {
        if ( v14 )
          goto LABEL_11;
        v21 = 0;
LABEL_35:
        v22 = v11 & 0xFFFFFFFFFFFFFFF8LL;
        goto LABEL_21;
      }
      if ( v14 != (v11 & 0xFFFFFFFFFFFFFFF8LL) )
        goto LABEL_11;
LABEL_22:
      if ( v10[3] != v9[3]
        || v10[1] != v9[1]
        || *((_WORD *)v10 + 16) != *((_WORD *)v9 + 16)
        || v9[6] != v10[6]
        || v9[5] != v10[5]
        || v9[7] != v10[7]
        || v10[8] != v9[8]
        || (v24 = v8, v25 = *v6, v26 = sub_1E34390(*v6), v26 != sub_1E34390(v9))
        || (v27 = sub_1E340A0(v25), v23 = sub_1E340A0(v9), v8 = v24, v27 != v23) )
      {
LABEL_11:
        v2 = *(_BYTE *)(a1 + 49);
        v5 = *(_BYTE *)(a2 + 49);
        goto LABEL_12;
      }
      ++v6;
      ++v7;
      if ( v24 == v6 )
        return *(char **)(a1 + 56);
    }
    if ( (_BYTE)v13 )
    {
      v21 = v12 & 0xFFFFFFFFFFFFFFF8LL;
      goto LABEL_35;
    }
    if ( (v11 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      goto LABEL_11;
    v21 = v12 & 0xFFFFFFFFFFFFFFF8LL;
    v22 = 0;
LABEL_21:
    if ( v22 != v21 )
      goto LABEL_11;
    goto LABEL_22;
  }
LABEL_12:
  v15 = v5 + v2;
  if ( v15 != (unsigned __int64)(unsigned __int8)(v5 + v2) )
    return 0;
  v16 = sub_1E15F70(a1);
  v17 = (char *)sub_1E0A240(v16, v15);
  v18 = 8LL * *(unsigned __int8 *)(a1 + 49);
  v19 = v18;
  if ( v18 )
    memmove(v17, *(const void **)(a1 + 56), v18);
  v20 = 8LL * *(unsigned __int8 *)(a2 + 49);
  if ( v20 )
    memmove(&v17[v19], *(const void **)(a2 + 56), v20);
  return v17;
}

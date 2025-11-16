// Function: sub_E41A50
// Address: 0xe41a50
//
__int64 __fastcall sub_E41A50(char *a1, size_t a2, __int64 *a3)
{
  size_t v3; // r12
  _BYTE *v5; // rax
  size_t v6; // rax
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rax
  size_t v11; // rdx
  size_t v13; // rdx
  __int64 v14; // rcx
  size_t v15; // rdx
  size_t v16; // rdx
  size_t v17; // rdx
  size_t v18; // rdx
  size_t v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-38h]

  v3 = a2;
  if ( a2 )
  {
    v5 = memchr(a1, 60, a2);
    if ( v5 )
    {
      v6 = v5 - a1;
      if ( v6 != -1 && a2 > v6 )
        v3 = v6;
    }
  }
  v7 = a3[1];
  v8 = *a3;
  v9 = (v7 - v8) >> 6;
  v10 = (v7 - v8) >> 4;
  if ( v9 > 0 )
  {
    v9 = v8 + (v9 << 6);
    while ( 1 )
    {
      v11 = *(_QWORD *)(v8 + 8);
      if ( v3 >= v11 && (!v11 || !memcmp(&a1[v3 - v11], *(const void **)v8, v11)) )
        goto LABEL_11;
      v13 = *(_QWORD *)(v8 + 24);
      if ( v3 >= v13 )
      {
        v14 = v8 + 16;
        if ( !v13 )
          goto LABEL_17;
        v20 = v8 + 16;
        if ( !memcmp(&a1[v3 - v13], *(const void **)(v8 + 16), v13) )
          goto LABEL_16;
      }
      v15 = *(_QWORD *)(v8 + 40);
      if ( v3 >= v15 )
      {
        v14 = v8 + 32;
        if ( !v15 )
          goto LABEL_17;
        v20 = v8 + 32;
        if ( !memcmp(&a1[v3 - v15], *(const void **)(v8 + 32), v15) )
          goto LABEL_16;
      }
      v16 = *(_QWORD *)(v8 + 56);
      if ( v3 >= v16 )
      {
        v14 = v8 + 48;
        if ( !v16 )
          goto LABEL_17;
        v20 = v8 + 48;
        if ( !memcmp(&a1[v3 - v16], *(const void **)(v8 + 48), v16) )
        {
LABEL_16:
          v14 = v20;
LABEL_17:
          LOBYTE(v9) = v7 != v14;
          return (unsigned int)v9;
        }
      }
      v8 += 64;
      if ( v8 == v9 )
      {
        v10 = (v7 - v8) >> 4;
        break;
      }
    }
  }
  if ( v10 != 2 )
  {
    if ( v10 != 3 )
    {
      LODWORD(v9) = 0;
      if ( v10 != 1 )
        return (unsigned int)v9;
      goto LABEL_29;
    }
    v18 = *(_QWORD *)(v8 + 8);
    if ( v18 <= v3 && (!v18 || !memcmp(&a1[v3 - v18], *(const void **)v8, v18)) )
      goto LABEL_11;
    v8 += 16;
  }
  v19 = *(_QWORD *)(v8 + 8);
  if ( v3 >= v19 && (!v19 || !memcmp(&a1[v3 - v19], *(const void **)v8, v19)) )
    goto LABEL_11;
  v8 += 16;
LABEL_29:
  v17 = *(_QWORD *)(v8 + 8);
  LODWORD(v9) = 0;
  if ( v17 <= v3 && (!v17 || !memcmp(&a1[v3 - v17], *(const void **)v8, v17)) )
LABEL_11:
    LOBYTE(v9) = v7 != v8;
  return (unsigned int)v9;
}

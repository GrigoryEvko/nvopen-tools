// Function: sub_27ECDA0
// Address: 0x27ecda0
//
__int64 *__fastcall sub_27ECDA0(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v3; // rdi
  __int64 *result; // rax
  __int64 *v5; // rdx
  __int64 v6; // rax
  _QWORD **v7; // r12
  __int64 *v8; // r15
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 *v12; // r13
  unsigned __int64 v13; // rax
  __int64 *v14; // r14
  char v15; // al
  __int64 v16; // rdx
  char v17; // al
  __int64 v18; // rdx
  char v19; // al
  __int64 v20; // rdx
  char v21; // al
  __int64 *i; // r13
  unsigned __int64 v23; // rax
  char v24; // al
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rcx
  char v29; // al
  unsigned __int64 v30; // rax
  char v31; // al
  unsigned __int64 v32; // rax
  char v33; // al
  unsigned __int64 v34; // rax
  __int64 v35; // [rsp+0h] [rbp-40h]
  __int64 *src; // [rsp+8h] [rbp-38h]

  v3 = *a1;
  if ( !*(_BYTE *)(v3 + 28) )
  {
    result = sub_C8CA60(v3, (__int64)a2);
    if ( result )
      return result;
LABEL_8:
    v6 = a1[1];
    v7 = (_QWORD **)a1[2];
    v8 = *(__int64 **)v6;
    v35 = v6;
    v9 = 8LL * *(unsigned int *)(v6 + 8);
    src = (__int64 *)(*(_QWORD *)v6 + v9);
    v10 = v9 >> 3;
    v11 = v9 >> 5;
    if ( v11 )
    {
      v12 = &v8[4 * v11];
      while ( 1 )
      {
        v21 = sub_FD60A0(*v8 & 0xFFFFFFFFFFFFFFF8LL, a2, v7);
        if ( (v21 & 2) != 0 )
          goto LABEL_23;
        if ( (v21 & 1) != 0 )
        {
          v13 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
          *v8 |= 4uLL;
          if ( (*(_BYTE *)(v13 + 67) & 0x10) == 0 )
            goto LABEL_23;
        }
        v14 = v8 + 1;
        v15 = sub_FD60A0(v8[1] & 0xFFFFFFFFFFFFFFF8LL, a2, v7);
        if ( (v15 & 2) != 0 )
          goto LABEL_34;
        if ( (v15 & 1) != 0 )
        {
          v16 = v8[1];
          v8[1] = v16 | 4;
          if ( (*(_BYTE *)((v16 & 0xFFFFFFFFFFFFFFF8LL) + 67) & 0x10) == 0 )
            goto LABEL_34;
        }
        v14 = v8 + 2;
        v17 = sub_FD60A0(v8[2] & 0xFFFFFFFFFFFFFFF8LL, a2, v7);
        if ( (v17 & 2) != 0 )
          goto LABEL_34;
        if ( (v17 & 1) != 0 )
        {
          v18 = v8[2];
          v8[2] = v18 | 4;
          if ( (*(_BYTE *)((v18 & 0xFFFFFFFFFFFFFFF8LL) + 67) & 0x10) == 0 )
            goto LABEL_34;
        }
        v14 = v8 + 3;
        v19 = sub_FD60A0(v8[3] & 0xFFFFFFFFFFFFFFF8LL, a2, v7);
        if ( (v19 & 2) != 0
          || (v19 & 1) != 0
          && (v20 = v8[3], v8[3] = v20 | 4, (*(_BYTE *)((v20 & 0xFFFFFFFFFFFFFFF8LL) + 67) & 0x10) == 0) )
        {
LABEL_34:
          v8 = v14;
          goto LABEL_23;
        }
        v8 += 4;
        if ( v12 == v8 )
        {
          v10 = src - v8;
          break;
        }
      }
    }
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 != 1 )
        {
LABEL_39:
          v8 = src;
LABEL_31:
          v26 = *(_QWORD *)v35;
          v27 = *(_QWORD *)v35 + 8LL * *(unsigned int *)(v35 + 8) - (_QWORD)src;
          if ( src != (__int64 *)(*(_QWORD *)v35 + 8LL * *(unsigned int *)(v35 + 8)) )
          {
            memmove(v8, src, *(_QWORD *)v35 + 8LL * *(unsigned int *)(v35 + 8) - (_QWORD)src);
            v26 = *(_QWORD *)v35;
          }
          v28 = (__int64)v8 + v27 - v26;
          *(_DWORD *)(v35 + 8) = v28 >> 3;
          return (__int64 *)v35;
        }
LABEL_48:
        v33 = sub_FD60A0(*v8 & 0xFFFFFFFFFFFFFFF8LL, a2, v7);
        if ( (v33 & 2) == 0 )
        {
          if ( (v33 & 1) == 0 )
            goto LABEL_39;
          v34 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
          *v8 |= 4uLL;
          if ( (*(_BYTE *)(v34 + 67) & 0x10) != 0 )
            goto LABEL_39;
        }
LABEL_23:
        if ( src != v8 )
        {
          for ( i = v8 + 1; src != i; *(v8 - 1) = v25 )
          {
            while ( 1 )
            {
              v24 = sub_FD60A0(*i & 0xFFFFFFFFFFFFFFF8LL, a2, v7);
              if ( (v24 & 2) == 0 )
              {
                if ( (v24 & 1) == 0 )
                  break;
                v23 = *i & 0xFFFFFFFFFFFFFFF8LL;
                *i |= 4uLL;
                if ( (*(_BYTE *)(v23 + 67) & 0x10) != 0 )
                  break;
              }
              if ( src == ++i )
                goto LABEL_31;
            }
            v25 = *i;
            ++v8;
            ++i;
          }
        }
        goto LABEL_31;
      }
      v29 = sub_FD60A0(*v8 & 0xFFFFFFFFFFFFFFF8LL, a2, v7);
      if ( (v29 & 2) != 0 )
        goto LABEL_23;
      if ( (v29 & 1) != 0 )
      {
        v30 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
        *v8 |= 4uLL;
        if ( (*(_BYTE *)(v30 + 67) & 0x10) == 0 )
          goto LABEL_23;
      }
      ++v8;
    }
    v31 = sub_FD60A0(*v8 & 0xFFFFFFFFFFFFFFF8LL, a2, v7);
    if ( (v31 & 2) != 0 )
      goto LABEL_23;
    if ( (v31 & 1) != 0 )
    {
      v32 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
      *v8 |= 4uLL;
      if ( (*(_BYTE *)(v32 + 67) & 0x10) == 0 )
        goto LABEL_23;
    }
    ++v8;
    goto LABEL_48;
  }
  result = *(__int64 **)(v3 + 8);
  v5 = &result[*(unsigned int *)(v3 + 20)];
  if ( result == v5 )
    goto LABEL_8;
  while ( a2 != (unsigned __int8 *)*result )
  {
    if ( v5 == ++result )
      goto LABEL_8;
  }
  return result;
}

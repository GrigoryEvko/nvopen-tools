// Function: sub_17B7590
// Address: 0x17b7590
//
void __fastcall sub_17B7590(size_t **a1, size_t **a2)
{
  int v2; // eax
  size_t v3; // r8
  size_t **v4; // r13
  size_t *v5; // r15
  const void *v6; // rsi
  size_t v7; // r12
  const void *v8; // r13
  size_t **i; // r14
  int v10; // eax
  size_t v11; // r10
  size_t *v12; // rbx
  const void *v13; // rsi
  size_t **v14; // r13
  size_t **v16; // [rsp+18h] [rbp-48h]
  size_t v17; // [rsp+20h] [rbp-40h]
  size_t v18; // [rsp+20h] [rbp-40h]
  size_t v19; // [rsp+20h] [rbp-40h]
  size_t v20; // [rsp+20h] [rbp-40h]
  size_t **v21; // [rsp+28h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 1 )
  {
    v16 = a1 + 1;
    v21 = a1 + 2;
    do
    {
      while ( 1 )
      {
        v5 = *v16;
        v3 = **a1;
        v6 = *a1 + 22;
        v7 = **v16;
        v8 = *v16 + 22;
        if ( v3 >= v7 )
          break;
        if ( !v3 )
          goto LABEL_15;
        v18 = **a1;
        v2 = memcmp(*v16 + 22, v6, v18);
        v3 = v18;
        if ( v2 )
          goto LABEL_14;
LABEL_7:
        if ( v3 > v7 )
          goto LABEL_8;
LABEL_15:
        for ( i = v16; ; --i )
        {
          v12 = *(i - 1);
          v11 = *v12;
          v13 = v12 + 22;
          if ( v7 > *v12 )
            break;
          if ( v7 )
          {
            v19 = *v12;
            v10 = memcmp(v8, v13, v7);
            v11 = v19;
            if ( v10 )
              goto LABEL_24;
          }
          if ( v7 == v11 )
            goto LABEL_25;
LABEL_19:
          if ( v7 >= v11 )
            goto LABEL_25;
LABEL_20:
          *i = v12;
          v7 = *v5;
        }
        if ( !v11 )
          goto LABEL_25;
        v20 = *v12;
        v10 = memcmp(v8, v13, *v12);
        v11 = v20;
        if ( !v10 )
          goto LABEL_19;
LABEL_24:
        if ( v10 < 0 )
          goto LABEL_20;
LABEL_25:
        v14 = v21;
        ++v16;
        ++v21;
        *i = v5;
        if ( a2 == v14 )
          return;
      }
      if ( !v7 || (v17 = **a1, v2 = memcmp(*v16 + 22, v6, **v16), v3 = v17, !v2) )
      {
        if ( v3 == v7 )
          goto LABEL_15;
        goto LABEL_7;
      }
LABEL_14:
      if ( v2 >= 0 )
        goto LABEL_15;
LABEL_8:
      v4 = v21;
      if ( a1 != v16 )
        memmove(a1 + 1, a1, (char *)v16 - (char *)a1);
      ++v16;
      ++v21;
      *a1 = v5;
    }
    while ( a2 != v4 );
  }
}

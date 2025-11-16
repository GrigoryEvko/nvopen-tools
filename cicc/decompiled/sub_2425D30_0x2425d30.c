// Function: sub_2425D30
// Address: 0x2425d30
//
void __fastcall sub_2425D30(size_t **a1, size_t **a2)
{
  size_t **v2; // r13
  size_t *v3; // r14
  size_t v4; // r15
  size_t v5; // r12
  const void *v6; // r13
  size_t v7; // rdx
  int v8; // eax
  size_t **i; // r15
  size_t *v10; // rbx
  size_t v11; // r9
  size_t v12; // rdx
  int v13; // eax
  size_t **v14; // r13
  size_t **v15; // [rsp+18h] [rbp-48h]
  size_t **v16; // [rsp+20h] [rbp-40h]
  size_t v17; // [rsp+28h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 1 )
  {
    v15 = a1 + 1;
    v16 = a1 + 2;
    while ( 1 )
    {
      v3 = *v15;
      v4 = **a1;
      v5 = **v15;
      v6 = *v15 + 24;
      v7 = v5;
      if ( v4 <= v5 )
        v7 = **a1;
      if ( v7 && (v8 = memcmp(*v15 + 24, *a1 + 24, v7)) != 0 )
      {
        if ( v8 < 0 )
          goto LABEL_4;
LABEL_13:
        for ( i = v15; ; --i )
        {
          v10 = *(i - 1);
          v11 = *v10;
          v12 = *v10;
          if ( v5 <= *v10 )
            v12 = v5;
          if ( !v12 )
            break;
          v17 = *v10;
          v13 = memcmp(v6, v10 + 24, v12);
          v11 = v17;
          if ( !v13 )
            break;
          if ( v13 >= 0 )
            goto LABEL_20;
LABEL_23:
          *i = v10;
          v5 = *v3;
        }
        if ( v5 != v11 && v5 < v11 )
          goto LABEL_23;
LABEL_20:
        v14 = v16;
        ++v15;
        ++v16;
        *i = v3;
        if ( a2 == v14 )
          return;
      }
      else
      {
        if ( v4 == v5 || v4 <= v5 )
          goto LABEL_13;
LABEL_4:
        v2 = v16;
        if ( a1 != v15 )
          memmove(a1 + 1, a1, (char *)v15 - (char *)a1);
        ++v15;
        ++v16;
        *a1 = v3;
        if ( a2 == v2 )
          return;
      }
    }
  }
}

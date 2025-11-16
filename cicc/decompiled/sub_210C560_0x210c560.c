// Function: sub_210C560
// Address: 0x210c560
//
void __fastcall sub_210C560(__int64 **a1, __int64 **a2)
{
  __int64 **v2; // rbx
  int v3; // eax
  __int64 *v4; // r14
  __int64 **v5; // r12
  __int64 *v6; // r12
  const char *v7; // r15
  size_t v8; // rdx
  size_t v9; // r14
  size_t v10; // rdx
  const char *v11; // rdi
  size_t v12; // r12
  __int64 **v13; // r14
  int v14; // eax
  __int64 *v15; // rax
  const char *v16; // r15
  size_t v17; // rdx
  size_t v18; // r13
  size_t v19; // rdx
  const char *v20; // rdi
  size_t v21; // r12
  __int64 *v23; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 )
  {
    v2 = a1 + 1;
    if ( a2 != a1 + 1 )
    {
      do
      {
        while ( 1 )
        {
          v6 = *v2;
          v7 = sub_1649960(**a1);
          v9 = v8;
          v11 = sub_1649960(*v6);
          v12 = v10;
          if ( v10 <= v9 )
            break;
          if ( !v9 )
            goto LABEL_15;
          v3 = memcmp(v11, v7, v9);
          if ( v3 )
            goto LABEL_14;
LABEL_7:
          if ( v12 < v9 )
            goto LABEL_8;
LABEL_15:
          v13 = v2;
          v23 = *v2;
          while ( 1 )
          {
            v16 = sub_1649960(**(v13 - 1));
            v18 = v17;
            v20 = sub_1649960(*v23);
            v21 = v19;
            if ( v19 > v18 )
              break;
            if ( v19 )
            {
              v14 = memcmp(v20, v16, v19);
              if ( v14 )
                goto LABEL_24;
            }
            if ( v21 == v18 )
              goto LABEL_25;
LABEL_19:
            if ( v21 >= v18 )
              goto LABEL_25;
LABEL_20:
            v15 = *--v13;
            v13[1] = v15;
          }
          if ( !v18 )
            goto LABEL_25;
          v14 = memcmp(v20, v16, v18);
          if ( !v14 )
            goto LABEL_19;
LABEL_24:
          if ( v14 < 0 )
            goto LABEL_20;
LABEL_25:
          ++v2;
          *v13 = v23;
          if ( a2 == v2 )
            return;
        }
        if ( !v10 || (v3 = memcmp(v11, v7, v10)) == 0 )
        {
          if ( v12 == v9 )
            goto LABEL_15;
          goto LABEL_7;
        }
LABEL_14:
        if ( v3 >= 0 )
          goto LABEL_15;
LABEL_8:
        v4 = *v2;
        v5 = v2 + 1;
        if ( a1 != v2 )
          memmove(a1 + 1, a1, (char *)v2 - (char *)a1);
        ++v2;
        *a1 = v4;
      }
      while ( a2 != v5 );
    }
  }
}

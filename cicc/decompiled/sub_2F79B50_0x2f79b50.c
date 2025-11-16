// Function: sub_2F79B50
// Address: 0x2f79b50
//
void __fastcall sub_2F79B50(__int64 **a1, __int64 **a2)
{
  __int64 **v2; // r12
  __int64 *v3; // r13
  __int64 **v4; // rbx
  __int64 *v5; // r13
  const char *v6; // rax
  size_t v7; // rdx
  size_t v8; // rbx
  const char *v9; // r15
  const char *v10; // rax
  size_t v11; // rdx
  size_t v12; // r13
  bool v13; // cc
  size_t v14; // rdx
  int v15; // eax
  __int64 **v16; // r15
  const char *v17; // r13
  size_t v18; // rdx
  size_t v19; // rbx
  const char *v20; // rax
  size_t v21; // rdx
  size_t v22; // r14
  int v23; // eax
  __int64 *v24; // rax
  __int64 *v25; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 )
  {
    v2 = a1 + 1;
    if ( a2 != a1 + 1 )
    {
      while ( 1 )
      {
        v5 = *v2;
        v6 = sub_BD5D20(**a1);
        v8 = v7;
        v9 = v6;
        v10 = sub_BD5D20(*v5);
        v12 = v11;
        v13 = v11 <= v8;
        v14 = v8;
        if ( v13 )
          v14 = v12;
        if ( v14 && (v15 = memcmp(v10, v9, v14)) != 0 )
        {
          if ( v15 < 0 )
            goto LABEL_4;
LABEL_13:
          v16 = v2;
          v25 = *v2;
          while ( 1 )
          {
            v17 = sub_BD5D20(**(v16 - 1));
            v19 = v18;
            v20 = sub_BD5D20(*v25);
            v22 = v21;
            if ( v19 <= v21 )
              v21 = v19;
            if ( !v21 )
              break;
            v23 = memcmp(v20, v17, v21);
            if ( !v23 )
              break;
            if ( v23 >= 0 )
              goto LABEL_20;
LABEL_23:
            v24 = *--v16;
            v16[1] = v24;
          }
          if ( v19 != v22 && v19 > v22 )
            goto LABEL_23;
LABEL_20:
          ++v2;
          *v16 = v25;
          if ( v2 == a2 )
            return;
        }
        else
        {
          if ( v12 == v8 || v12 >= v8 )
            goto LABEL_13;
LABEL_4:
          v3 = *v2;
          v4 = v2 + 1;
          if ( a1 != v2 )
            memmove(a1 + 1, a1, (char *)v2 - (char *)a1);
          ++v2;
          *a1 = v3;
          if ( v4 == a2 )
            return;
        }
      }
    }
  }
}

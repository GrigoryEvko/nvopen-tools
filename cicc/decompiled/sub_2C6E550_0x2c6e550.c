// Function: sub_2C6E550
// Address: 0x2c6e550
//
void __fastcall sub_2C6E550(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 *v3; // rbx
  __int64 v4; // r13
  const char *v5; // rax
  size_t v6; // rdx
  size_t v7; // rbx
  const char *v8; // r15
  const char *v9; // rax
  size_t v10; // rdx
  size_t v11; // r13
  bool v12; // cc
  size_t v13; // rdx
  int v14; // eax
  __int64 *v15; // r15
  __int64 v16; // r13
  const char *v17; // rax
  size_t v18; // rdx
  size_t v19; // rbx
  const char *v20; // r12
  const char *v21; // rax
  size_t v22; // rdx
  size_t v23; // r14
  size_t v24; // rdx
  int v25; // eax
  __int64 v26; // rax
  __int64 *v27; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 && a1 + 1 != a2 )
  {
    v27 = a1 + 1;
    while ( 1 )
    {
      v4 = *v27;
      v5 = sub_BD5D20(*a1);
      v7 = v6;
      v8 = v5;
      v9 = sub_BD5D20(v4);
      v11 = v10;
      v12 = v10 <= v7;
      v13 = v7;
      if ( v12 )
        v13 = v11;
      if ( v13 && (v14 = memcmp(v9, v8, v13)) != 0 )
      {
        if ( v14 < 0 )
          goto LABEL_4;
LABEL_13:
        v15 = v27;
        v16 = *v27;
        while ( 1 )
        {
          v17 = sub_BD5D20(*(v15 - 1));
          v19 = v18;
          v20 = v17;
          v21 = sub_BD5D20(v16);
          v23 = v22;
          v12 = v22 <= v19;
          v24 = v19;
          if ( v12 )
            v24 = v23;
          if ( !v24 )
            break;
          v25 = memcmp(v21, v20, v24);
          if ( !v25 )
            break;
          if ( v25 >= 0 )
            goto LABEL_20;
LABEL_23:
          v26 = *--v15;
          v15[1] = v26;
        }
        if ( v23 != v19 && v23 < v19 )
          goto LABEL_23;
LABEL_20:
        *v15 = v16;
        if ( ++v27 == a2 )
          return;
      }
      else
      {
        if ( v11 == v7 || v11 >= v7 )
          goto LABEL_13;
LABEL_4:
        v2 = *v27;
        v3 = v27 + 1;
        if ( a1 != v27 )
          memmove(a1 + 1, a1, (char *)v27 - (char *)a1);
        ++v27;
        *a1 = v2;
        if ( v3 == a2 )
          return;
      }
    }
  }
}

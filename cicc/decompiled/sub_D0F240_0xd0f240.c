// Function: sub_D0F240
// Address: 0xd0f240
//
void __fastcall sub_D0F240(char *src, char *a2)
{
  char *i; // r12
  const char *v4; // rax
  size_t v5; // rdx
  size_t v6; // rbx
  const char *v7; // r15
  const char *v8; // rax
  size_t v9; // rdx
  size_t v10; // r14
  bool v11; // cc
  size_t v12; // rdx
  int v13; // eax
  __int64 v14; // rbx
  char *v15; // r15
  __int64 v16; // rdi
  __int64 v17; // r8
  __int64 v18; // r14
  char *j; // r15
  __int64 v20; // rax
  __int64 v21; // rdi
  const char *v22; // rax
  size_t v23; // rdx
  size_t v24; // rbx
  const char *v25; // rax
  size_t v26; // rdx
  size_t v27; // r8
  int v28; // eax
  const char *s2; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  __int64 v31; // [rsp+18h] [rbp-38h]
  size_t v32; // [rsp+18h] [rbp-38h]

  if ( src != a2 )
  {
    for ( i = src + 8; a2 != i; i += 8 )
    {
      while ( 1 )
      {
        v14 = *(_QWORD *)i;
        v16 = *(_QWORD *)(*(_QWORD *)src + 8LL);
        v17 = *(_QWORD *)(*(_QWORD *)i + 8LL);
        v18 = *(_QWORD *)i;
        if ( !v16 )
          goto LABEL_16;
        if ( v17 )
          break;
LABEL_12:
        v15 = i + 8;
        if ( src != i )
          memmove(src + 8, src, i - src);
        *(_QWORD *)src = v14;
        i += 8;
        if ( a2 == v15 )
          return;
      }
      v30 = *(_QWORD *)(*(_QWORD *)i + 8LL);
      v4 = sub_BD5D20(v16);
      v6 = v5;
      v7 = v4;
      v8 = sub_BD5D20(v30);
      v10 = v9;
      v11 = v9 <= v6;
      v12 = v6;
      if ( v11 )
        v12 = v10;
      if ( v12 && (v13 = memcmp(v8, v7, v12)) != 0 )
      {
        if ( v13 < 0 )
          goto LABEL_11;
      }
      else if ( v10 != v6 && v10 < v6 )
      {
LABEL_11:
        v14 = *(_QWORD *)i;
        goto LABEL_12;
      }
      v18 = *(_QWORD *)i;
      v17 = *(_QWORD *)(*(_QWORD *)i + 8LL);
LABEL_16:
      for ( j = i; ; j -= 8 )
      {
        v20 = *((_QWORD *)j - 1);
        v21 = *(_QWORD *)(v20 + 8);
        if ( !v21 )
          break;
        v31 = v17;
        if ( v17 )
        {
          v22 = sub_BD5D20(v21);
          v24 = v23;
          s2 = v22;
          v25 = sub_BD5D20(v31);
          v27 = v26;
          if ( v24 <= v26 )
            v26 = v24;
          if ( v26 && (v32 = v27, v28 = memcmp(v25, s2, v26), v27 = v32, v28) )
          {
            if ( v28 >= 0 )
              break;
          }
          else if ( v24 == v27 || v24 <= v27 )
          {
            break;
          }
          v20 = *((_QWORD *)j - 1);
        }
        *(_QWORD *)j = v20;
        v17 = *(_QWORD *)(v18 + 8);
      }
      *(_QWORD *)j = v18;
    }
  }
}

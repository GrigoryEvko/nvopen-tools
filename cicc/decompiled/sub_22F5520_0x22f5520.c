// Function: sub_22F5520
// Address: 0x22f5520
//
__int64 __fastcall sub_22F5520(void **a1, __int64 a2, unsigned int *a3, char *a4, size_t a5, char a6)
{
  __int64 v8; // rsi
  unsigned int v9; // ebx
  char *v10; // rdx
  size_t v11; // r8
  const char *v12; // rcx
  size_t v13; // rax
  size_t v14; // rbx
  unsigned __int64 v15; // rbx
  unsigned int *v16; // rcx
  unsigned int *v17; // r12
  size_t v18; // rax
  size_t v19; // r15
  __int64 v20; // rsi
  bool v21; // zf
  const char *v22; // rdx
  const char *v23; // r14
  char *v24; // rdi
  unsigned __int64 v25; // rax
  const char *v27; // rax
  const char *v28; // rdi
  unsigned int v29; // eax
  __int64 v30; // rcx
  char *v31; // [rsp+8h] [rbp-78h]
  size_t v32; // [rsp+10h] [rbp-70h]
  const char *s2; // [rsp+18h] [rbp-68h]
  char *s2a; // [rsp+18h] [rbp-68h]
  const char *v37; // [rsp+30h] [rbp-50h]
  unsigned int *v38; // [rsp+30h] [rbp-50h]
  char *v40; // [rsp+40h] [rbp-40h] BYREF
  size_t v41; // [rsp+48h] [rbp-38h]

  v8 = *a3;
  v9 = a3[1];
  v10 = (char *)*a1;
  if ( !(_DWORD)v8 )
  {
    v11 = 0;
    v12 = &v10[v9];
    if ( !v12 )
      return 0;
    goto LABEL_3;
  }
  v28 = &v10[*(unsigned int *)(a2 + 4LL * (unsigned int)(v8 + 1))];
  if ( !v28 )
  {
    v11 = 0;
    v12 = &v10[v9];
    if ( !v12 )
    {
      v14 = 0;
      goto LABEL_5;
    }
    goto LABEL_3;
  }
  s2a = v10;
  v29 = strlen(v28);
  v10 = s2a;
  v30 = v9;
  v14 = 0;
  v8 = (unsigned int)v8;
  v11 = v29;
  v12 = &s2a[v30];
  if ( v12 )
  {
LABEL_3:
    v31 = v10;
    v32 = v11;
    v37 = v12;
    v13 = strlen(v12);
    v10 = v31;
    v11 = v32;
    v8 = (unsigned int)v8;
    v12 = v37;
    v14 = v13;
  }
  if ( v11 > v14 )
  {
    v27 = &v12[v14];
    v15 = 0;
    s2 = v27;
    goto LABEL_6;
  }
LABEL_5:
  v15 = v14 - v11;
  s2 = &v12[v11];
LABEL_6:
  if ( (_DWORD)v8 )
  {
    v16 = (unsigned int *)(a2 + 4LL * (unsigned int)(v8 + 1));
    v38 = &v16[*(unsigned int *)(a2 + 4 * v8)];
    if ( v38 != v16 )
    {
      v17 = (unsigned int *)(a2 + 4LL * (unsigned int)(v8 + 1));
      while ( 1 )
      {
        v20 = *v17;
        v21 = &v10[v20] == 0;
        v22 = &v10[v20];
        v23 = v22;
        if ( v21 )
          goto LABEL_13;
        v18 = strlen(v22);
        v19 = v18;
        if ( v18 > a5 )
          goto LABEL_10;
        if ( !v18 )
        {
LABEL_13:
          v24 = a4;
          v25 = a5;
          LODWORD(v19) = 0;
          v40 = a4;
          v41 = a5;
          if ( a6 )
            goto LABEL_23;
        }
        else
        {
          if ( memcmp(a4, v23, v18) )
            goto LABEL_10;
          v24 = &a4[v19];
          v40 = &a4[v19];
          v25 = a5 - v19;
          v41 = a5 - v19;
          if ( a6 )
          {
LABEL_23:
            if ( (unsigned __int8)sub_C92F10(&v40, (__int64)s2, v15) )
              return (unsigned int)(v19 + v15);
            goto LABEL_10;
          }
        }
        if ( v15 <= v25 && (!v15 || !memcmp(v24, s2, v15)) )
          return (unsigned int)(v19 + v15);
LABEL_10:
        if ( v38 == ++v17 )
          return 0;
        v10 = (char *)*a1;
      }
    }
  }
  return 0;
}

// Function: sub_CD93F0
// Address: 0xcd93f0
//
void __fastcall sub_CD93F0(__int64 *a1, size_t a2)
{
  __int64 v3; // rcx
  size_t v4; // r12
  size_t v5; // rax
  bool v6; // cf
  __int64 v7; // rax
  __int64 v8; // r15
  char *v9; // r15
  char *v10; // r14
  __int64 v11; // r8
  __int64 v12; // rsi
  __int64 v13; // [rsp-40h] [rbp-40h]

  if ( a2 )
  {
    v3 = a1[1];
    v4 = v3 - *a1;
    if ( a2 <= a1[2] - v3 )
    {
      a1[1] = (__int64)memset((void *)a1[1], 0, a2) + a2;
      return;
    }
    if ( 0x7FFFFFFFFFFFFFFFLL - v4 < a2 )
      sub_4262D8((__int64)"vector::_M_default_append");
    v5 = v3 - *a1;
    if ( a2 >= v4 )
      v5 = a2;
    v6 = __CFADD__(v4, v5);
    v7 = v4 + v5;
    v8 = v7;
    if ( v6 || v7 < 0 )
    {
      v8 = 0x7FFFFFFFFFFFFFFFLL;
    }
    else if ( !v7 )
    {
      v9 = 0;
      v10 = 0;
      goto LABEL_11;
    }
    v10 = (char *)sub_22077B0(v8);
    v9 = &v10[v8];
LABEL_11:
    memset(&v10[v4], 0, a2);
    v11 = *a1;
    if ( a1[1] - *a1 > 0 )
    {
      v13 = *a1;
      memmove(v10, (const void *)*a1, a1[1] - *a1);
      v11 = v13;
      v12 = a1[2] - v13;
    }
    else
    {
      if ( !v11 )
      {
LABEL_13:
        *a1 = (__int64)v10;
        a1[2] = (__int64)v9;
        a1[1] = (__int64)&v10[v4 + a2];
        return;
      }
      v12 = a1[2] - v11;
    }
    j_j___libc_free_0(v11, v12);
    goto LABEL_13;
  }
}

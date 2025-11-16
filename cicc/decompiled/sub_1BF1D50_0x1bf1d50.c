// Function: sub_1BF1D50
// Address: 0x1bf1d50
//
__int64 __fastcall sub_1BF1D50(__int64 a1, __int64 a2, const char **a3, __int64 a4)
{
  _BYTE *v4; // r15
  __int64 v5; // rcx
  const char **v6; // rbx
  const char *v7; // r13
  __int64 v8; // r14
  size_t v9; // rdx
  size_t v10; // r12
  size_t v11; // rax
  const char **v13; // [rsp+8h] [rbp-38h]

  v4 = *(_BYTE **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  if ( *v4 )
    return 0;
  v5 = 2 * a4;
  v6 = a3;
  v13 = &a3[v5];
  if ( &a3[v5] == a3 )
    return 0;
  while ( 1 )
  {
    v7 = *v6;
    v8 = sub_161E970((__int64)v4);
    v10 = v9;
    if ( !v7 )
      break;
    v11 = strlen(v7);
    if ( v10 >= v11 && (!v11 || !memcmp((const void *)(v10 - v11 + v8), v7, v11)) )
      break;
    v6 += 2;
    if ( v13 == v6 )
      return 0;
  }
  return 1;
}

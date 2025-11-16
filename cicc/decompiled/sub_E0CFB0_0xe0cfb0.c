// Function: sub_E0CFB0
// Address: 0xe0cfb0
//
__int64 __fastcall sub_E0CFB0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r12
  char *v3; // r13
  unsigned __int64 v4; // rbx
  __int64 v5; // rbx
  int v6; // r14d
  __int64 v8; // rbx
  char v9; // [rsp+Fh] [rbp-31h]

  v2 = *a1;
  if ( *a1 )
  {
    v3 = (char *)a1[1];
    v4 = 0;
    do
    {
      v6 = *v3;
      v9 = *v3;
      if ( !isalpha(v6) || v4 > 0x9D89D89D89D89D7LL )
        break;
      v5 = 26 * v4;
      if ( (unsigned __int8)(v9 - 97) <= 0x19u )
      {
        v8 = v6 - 97 + v5;
        if ( v8 <= 0 )
          break;
        *a2 = v8;
        ++a1[1];
        --*a1;
        return 1;
      }
      a1[1] = (__int64)++v3;
      v4 = v6 - 65 + v5;
      *a1 = --v2;
    }
    while ( v2 );
  }
  *a1 = 0;
  a1[1] = 0;
  return 0;
}

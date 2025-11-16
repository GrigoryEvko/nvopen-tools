// Function: sub_2F06550
// Address: 0x2f06550
//
__int64 __fastcall sub_2F06550(
        unsigned __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6)
{
  unsigned int i; // esi
  __int64 *v8; // rax
  __int64 *v9; // rcx
  __int64 v10; // rdi

  for ( i = 1; ; ++i )
  {
    v8 = *(__int64 **)(a1 + 40);
    v9 = &v8[2 * *(unsigned int *)(a1 + 48)];
    if ( v8 == v9 )
      break;
    while ( 1 )
    {
      v10 = *v8;
      if ( (((unsigned __int8)*v8 ^ 6) & 6) == 0 && *((_DWORD *)v8 + 2) == 5 )
        break;
      v8 += 2;
      if ( v9 == v8 )
        goto LABEL_10;
    }
    LOBYTE(v8) = i < a2;
    a6 = (unsigned int)v8;
    a1 = v10 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !a1 || i >= a2 )
      return a6;
  }
LABEL_10:
  LOBYTE(a6) = i < a2;
  return a6;
}

// Function: sub_2F92180
// Address: 0x2f92180
//
__int64 **__fastcall sub_2F92180(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 **v3; // rbx
  __int64 **result; // rax
  __int64 **i; // rbx
  __int64 **v6; // r15
  int v7; // r12d
  __int64 **v9; // [rsp+8h] [rbp-38h]

  v3 = *(__int64 ***)(a3 + 80);
  result = &v3[4 * *(unsigned int *)(a3 + 88)];
  v9 = result;
  if ( result != v3 )
  {
    for ( i = v3 + 1; ; i += 4 )
    {
      v6 = (__int64 **)*i;
      v7 = *(_DWORD *)(a3 + 228);
      if ( i != (__int64 **)*i )
      {
        do
        {
          sub_2F920F0(a1, a2, v6[2], v7);
          v6 = (__int64 **)*v6;
        }
        while ( i != v6 );
      }
      result = i + 4;
      if ( v9 == i + 3 )
        break;
    }
  }
  return result;
}

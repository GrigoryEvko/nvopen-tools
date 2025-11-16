// Function: sub_22076E0
// Address: 0x22076e0
//
unsigned __int64 __fastcall sub_22076E0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r9
  unsigned __int64 i; // rax
  __int64 v5; // r8
  int v6; // esi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rsi

  v3 = (__int64 *)((char *)a1 + (a2 & 0xFFFFFFFFFFFFFFF8LL));
  for ( i = a3 ^ (0xC6A4A7935BD1E995LL * a2);
        v3 != a1;
        i = 0xC6A4A7935BD1E995LL
          * (i ^ (0xC6A4A7935BD1E995LL * ((0xC6A4A7935BD1E995LL * v5) ^ ((0xC6A4A7935BD1E995LL * v5) >> 47)))) )
  {
    v5 = *a1++;
  }
  v6 = a2 & 7;
  if ( v6 )
  {
    v7 = 0;
    v8 = v6 - 1;
    do
    {
      v9 = *((unsigned __int8 *)v3 + v8--);
      v7 = v9 + (v7 << 8);
    }
    while ( (int)v8 >= 0 );
    i = 0xC6A4A7935BD1E995LL * (i ^ v7);
  }
  return ((0xC6A4A7935BD1E995LL * ((i >> 47) ^ i)) >> 47) ^ (0xC6A4A7935BD1E995LL * ((i >> 47) ^ i));
}

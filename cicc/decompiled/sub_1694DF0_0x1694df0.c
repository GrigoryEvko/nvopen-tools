// Function: sub_1694DF0
// Address: 0x1694df0
//
void __fastcall sub_1694DF0(unsigned int *a1, int a2)
{
  unsigned int *v2; // r14
  unsigned __int32 v3; // eax
  unsigned int v5; // r13d
  unsigned int v6; // esi
  unsigned __int8 *v7; // rax
  int v8; // edx
  int v9; // ecx
  __int64 v10; // rdx

  if ( a2 != 1 )
  {
    v2 = a1 + 2;
    *a1 = _byteswap_ulong(*a1);
    v3 = _byteswap_ulong(a1[1]);
    a1[1] = v3;
    if ( v3 )
    {
      v5 = 0;
      do
      {
        sub_1694C80(v2, a2, 1);
        v6 = v2[1];
        if ( v6 )
        {
          v7 = (unsigned __int8 *)(v2 + 2);
          v8 = 0;
          do
          {
            v9 = *v7++;
            v8 += v9;
          }
          while ( v7 != (unsigned __int8 *)((char *)v2 + v6 + 8) );
          v10 = ((v6 + 15) & 0xFFFFFFF8) + 16 * v8;
        }
        else
        {
          v10 = 8;
        }
        v2 = (unsigned int *)((char *)v2 + v10);
        ++v5;
      }
      while ( a1[1] > v5 );
    }
  }
}

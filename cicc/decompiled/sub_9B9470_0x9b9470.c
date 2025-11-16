// Function: sub_9B9470
// Address: 0x9b9470
//
__int64 __fastcall sub_9B9470(__int64 a1, int a2, int a3)
{
  int i; // r15d
  __int64 v4; // rax
  int v5; // r14d

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x1000000000LL;
  if ( a3 )
  {
    for ( i = 0; i != a3; ++i )
    {
      if ( a2 )
      {
        v4 = *(unsigned int *)(a1 + 8);
        v5 = 0;
        do
        {
          if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
          {
            sub_C8D5F0(a1, a1 + 16, v4 + 1, 4);
            v4 = *(unsigned int *)(a1 + 8);
          }
          ++v5;
          *(_DWORD *)(*(_QWORD *)a1 + 4 * v4) = i;
          v4 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
          *(_DWORD *)(a1 + 8) = v4;
        }
        while ( a2 != v5 );
      }
    }
  }
  return a1;
}

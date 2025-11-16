// Function: sub_9B9520
// Address: 0x9b9520
//
__int64 __fastcall sub_9B9520(__int64 a1, int a2, int a3)
{
  int i; // r13d
  __int64 v5; // rax
  int v6; // r14d
  int v7; // ebx
  int v9; // [rsp+Ch] [rbp-34h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x1000000000LL;
  if ( a2 )
  {
    for ( i = 0; i != a2; ++i )
    {
      if ( a3 )
      {
        v5 = *(unsigned int *)(a1 + 8);
        v6 = i;
        v7 = 0;
        do
        {
          if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
          {
            v9 = a3;
            sub_C8D5F0(a1, a1 + 16, v5 + 1, 4);
            v5 = *(unsigned int *)(a1 + 8);
            a3 = v9;
          }
          ++v7;
          *(_DWORD *)(*(_QWORD *)a1 + 4 * v5) = v6;
          v6 += a2;
          v5 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
          *(_DWORD *)(a1 + 8) = v5;
        }
        while ( a3 != v7 );
      }
    }
  }
  return a1;
}

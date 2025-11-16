// Function: sub_3544E90
// Address: 0x3544e90
//
unsigned __int64 __fastcall sub_3544E90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r14
  unsigned __int64 result; // rax
  unsigned int v10; // r13d
  __int64 i; // rdx
  char v12; // cl
  __int64 v13; // rdi
  __int64 v14; // r10
  __int64 v15; // r10
  __int64 v16; // rsi
  unsigned __int64 v17; // rdi
  __int64 v18; // r9

  v7 = *(unsigned int *)(a2 + 48);
  result = *(unsigned int *)(a3 + 8);
  v10 = *(_DWORD *)(a2 + 48);
  if ( v7 != result )
  {
    if ( v7 >= result )
    {
      if ( v7 > *(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), *(unsigned int *)(a2 + 48), 8u, a5, a6);
        result = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
        for ( i = *(_QWORD *)a3 + 8 * v7; i != result; result += 8LL )
        {
LABEL_5:
          if ( result )
            *(_QWORD *)result = 0;
        }
      }
      else
      {
        result = *(_QWORD *)a3 + 8 * result;
        i = *(_QWORD *)a3 + 8 * v7;
        if ( result != i )
          goto LABEL_5;
      }
      *(_DWORD *)(a3 + 8) = v7;
      v10 = *(_DWORD *)(a2 + 48);
      goto LABEL_9;
    }
    *(_DWORD *)(a3 + 8) = v7;
  }
LABEL_9:
  if ( v10 > 1 )
  {
    result = 8;
    v12 = 0;
    do
    {
      if ( !*(_QWORD *)(*(_QWORD *)(a2 + 32) + 4 * result + 24) )
      {
        v13 = 1LL << v12++;
        *(_QWORD *)(*(_QWORD *)a3 + result) = v13;
      }
      result += 8LL;
    }
    while ( 8LL * v10 != result );
    v14 = *(unsigned int *)(a2 + 48);
    if ( (unsigned int)v14 > 1 )
    {
      v15 = 8 * v14;
      v16 = 8;
      do
      {
        result = *(_QWORD *)(a2 + 32);
        v17 = result + 4 * v16;
        if ( *(_QWORD *)(v17 + 24) )
        {
          *(_QWORD *)(*(_QWORD *)a3 + v16) = 1LL << v12;
          for ( result = 0;
                *(_DWORD *)(v17 + 8) > (unsigned int)result;
                *(_QWORD *)(*(_QWORD *)a3 + v16) |= *(_QWORD *)(*(_QWORD *)a3
                                                              + 8LL * *(unsigned int *)(*(_QWORD *)(v17 + 24) + 4 * v18)) )
          {
            v18 = (unsigned int)result;
            result = (unsigned int)(result + 1);
          }
          ++v12;
        }
        v16 += 8;
      }
      while ( v16 != v15 );
    }
  }
  return result;
}

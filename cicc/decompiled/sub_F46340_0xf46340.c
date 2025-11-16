// Function: sub_F46340
// Address: 0xf46340
//
void __fastcall sub_F46340(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const void *v6; // r14
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax

  if ( a3 != a1 )
  {
    v6 = (const void *)(a5 + 16);
    v9 = a1;
    do
    {
      while ( 1 )
      {
        if ( !v9 )
          BUG();
        if ( *(_BYTE *)(v9 - 24) == 85 )
        {
          v10 = *(_QWORD *)(v9 - 56);
          if ( v10 )
          {
            if ( !*(_BYTE *)v10
              && *(_QWORD *)(v10 + 24) == *(_QWORD *)(v9 + 56)
              && (*(_BYTE *)(v10 + 33) & 0x20) != 0
              && *(_DWORD *)(v10 + 36) == 155 )
            {
              break;
            }
          }
        }
        v9 = *(_QWORD *)(v9 + 8);
        if ( a3 == v9 )
          return;
      }
      v11 = *(_QWORD *)(*(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) - 24) + 24LL);
      v12 = *(unsigned int *)(a5 + 8);
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
      {
        sub_C8D5F0(a5, v6, v12 + 1, 8u, a5, a6);
        v12 = *(unsigned int *)(a5 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a5 + 8 * v12) = v11;
      ++*(_DWORD *)(a5 + 8);
      v9 = *(_QWORD *)(v9 + 8);
    }
    while ( a3 != v9 );
  }
}

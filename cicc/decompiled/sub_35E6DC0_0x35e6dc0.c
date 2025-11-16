// Function: sub_35E6DC0
// Address: 0x35e6dc0
//
__int64 __fastcall sub_35E6DC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 *v14; // rcx
  unsigned __int64 v15; // rdx

  result = *(_QWORD *)(a1 + 24);
  v7 = *(_QWORD *)(result + 56);
  v8 = result + 48;
  if ( v7 != result + 48 )
  {
    v9 = *(unsigned int *)(a1 + 88);
    do
    {
      if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
      {
        sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v9 + 1, 8u, a5, a6);
        v9 = *(unsigned int *)(a1 + 88);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v9) = v7;
      v9 = (unsigned int)(*(_DWORD *)(a1 + 88) + 1);
      *(_DWORD *)(a1 + 88) = v9;
      v7 = *(_QWORD *)(v7 + 8);
    }
    while ( v8 != v7 );
    result = *(_QWORD *)(a1 + 24);
    v10 = *(_QWORD *)(result + 56);
    v11 = result + 48;
    if ( result + 48 != v10 )
    {
      while ( 1 )
      {
        if ( !v10 )
          BUG();
        v12 = v10;
        if ( (*(_BYTE *)v10 & 4) == 0 && (*(_BYTE *)(v10 + 44) & 8) != 0 )
        {
          do
            v12 = *(_QWORD *)(v12 + 8);
          while ( (*(_BYTE *)(v12 + 44) & 8) != 0 );
        }
        v13 = *(_QWORD *)(v12 + 8);
        sub_2FAD510(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL) + 32LL), v10);
        sub_2E31080(*(_QWORD *)(a1 + 24) + 40LL, v10);
        v14 = *(__int64 **)(v10 + 8);
        v15 = *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL;
        result = v15 | *v14 & 7;
        *v14 = result;
        *(_QWORD *)(v15 + 8) = v14;
        *(_QWORD *)(v10 + 8) = 0;
        *(_QWORD *)v10 &= 7uLL;
        if ( v11 == v13 )
          break;
        v10 = v13;
      }
    }
  }
  return result;
}

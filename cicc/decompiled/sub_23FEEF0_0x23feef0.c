// Function: sub_23FEEF0
// Address: 0x23feef0
//
__int64 __fastcall sub_23FEEF0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rdi

  result = *(_QWORD *)a1;
  v3 = 32LL * *(unsigned int *)(a1 + 8);
  if ( v3 )
  {
    v5 = a2 + v3;
    do
    {
      if ( a2 )
      {
        *(_DWORD *)(a2 + 24) = 0;
        *(_QWORD *)(a2 + 8) = 0;
        *(_DWORD *)(a2 + 16) = 0;
        *(_DWORD *)(a2 + 20) = 0;
        *(_QWORD *)a2 = 1;
        v6 = *(_QWORD *)(result + 8);
        ++*(_QWORD *)result;
        v7 = *(_QWORD *)(a2 + 8);
        *(_QWORD *)(a2 + 8) = v6;
        LODWORD(v6) = *(_DWORD *)(result + 16);
        *(_QWORD *)(result + 8) = v7;
        LODWORD(v7) = *(_DWORD *)(a2 + 16);
        *(_DWORD *)(a2 + 16) = v6;
        LODWORD(v6) = *(_DWORD *)(result + 20);
        *(_DWORD *)(result + 16) = v7;
        LODWORD(v7) = *(_DWORD *)(a2 + 20);
        *(_DWORD *)(a2 + 20) = v6;
        LODWORD(v6) = *(_DWORD *)(result + 24);
        *(_DWORD *)(result + 20) = v7;
        LODWORD(v7) = *(_DWORD *)(a2 + 24);
        *(_DWORD *)(a2 + 24) = v6;
        *(_DWORD *)(result + 24) = v7;
      }
      a2 += 32;
      result += 32;
    }
    while ( a2 != v5 );
    v8 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v10 = *(unsigned int *)(v9 - 8);
        v11 = *(_QWORD *)(v9 - 24);
        v9 -= 32;
        result = sub_C7D6A0(v11, 8 * v10, 8);
      }
      while ( v9 != v8 );
    }
  }
  return result;
}

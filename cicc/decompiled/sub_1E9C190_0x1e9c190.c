// Function: sub_1E9C190
// Address: 0x1e9c190
//
__int64 __fastcall sub_1E9C190(__int64 a1, int a2, unsigned int a3)
{
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  _BYTE *v8; // r9

  result = 0;
  if ( *(_DWORD *)(a1 + 16) == 1 )
  {
    sub_1E310D0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 40LL, a2);
    if ( a3 )
    {
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 40LL * (unsigned int)(*(_DWORD *)(a1 + 16) + 1) + 24) = a3;
    }
    else
    {
      *(_DWORD *)(a1 + 16) = -1;
      sub_1E16C90(*(_QWORD *)(a1 + 8), 2u, v5, v6, v7, v8);
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL) = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL) + 960LL;
    }
    return 1;
  }
  return result;
}

// Function: sub_1781890
// Address: 0x1781890
//
__int64 __fastcall sub_1781890(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // rax
  __int64 v6; // rax

  v2 = *(_QWORD *)(a2 + 8);
  v3 = 0;
  if ( v2 )
  {
    if ( !*(_QWORD *)(v2 + 8) && *(_BYTE *)(a2 + 16) == 78 )
    {
      v5 = *(_QWORD *)(a2 - 24);
      if ( !*(_BYTE *)(v5 + 16) && *(_DWORD *)(v5 + 36) == *(_DWORD *)a1 )
      {
        v6 = *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL)
                       + 24
                       * (*(unsigned int *)(a1 + 8)
                        - (unsigned __int64)(*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)));
        if ( v6 )
        {
          v3 = 1;
          **(_QWORD **)(a1 + 16) = v6;
        }
      }
    }
  }
  return v3;
}

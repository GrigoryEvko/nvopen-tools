// Function: sub_173FD20
// Address: 0x173fd20
//
__int64 __fastcall sub_173FD20(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx

  result = 0;
  if ( *(_BYTE *)(a2 + 16) == 78 )
  {
    v3 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v3 + 16) && *(_DWORD *)(v3 + 36) == *(_DWORD *)a1 )
    {
      v4 = *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL)
                     + 24
                     * (*(unsigned int *)(a1 + 8)
                      - (unsigned __int64)(*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)));
      if ( v4 )
      {
        **(_QWORD **)(a1 + 16) = v4;
        return 1;
      }
    }
  }
  return result;
}

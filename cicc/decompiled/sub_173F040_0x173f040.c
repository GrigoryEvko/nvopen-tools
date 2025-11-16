// Function: sub_173F040
// Address: 0x173f040
//
__int64 __fastcall sub_173F040(__int64 a1, __int64 a2)
{
  int v2; // eax
  unsigned int v3; // r8d
  int v5; // eax
  __int64 *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax

  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 > 0x17u )
  {
    v5 = v2 - 24;
  }
  else
  {
    v3 = 0;
    if ( (_BYTE)v2 != 5 )
      return v3;
    v5 = *(unsigned __int16 *)(a2 + 18);
  }
  v3 = 0;
  if ( v5 == 36 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v6 = *(__int64 **)(a2 - 8);
    else
      v6 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v7 = *v6;
    v3 = 0;
    if ( *(_BYTE *)(*v6 + 16) == 78 )
    {
      v8 = *(_QWORD *)(v7 - 24);
      if ( !*(_BYTE *)(v8 + 16) && *(_DWORD *)(v8 + 36) == *(_DWORD *)a1 )
      {
        v9 = *(_QWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL)
                       + 24
                       * (*(unsigned int *)(a1 + 8)
                        - (unsigned __int64)(*(_DWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)));
        if ( v9 )
        {
          v3 = 1;
          **(_QWORD **)(a1 + 16) = v9;
        }
      }
    }
  }
  return v3;
}

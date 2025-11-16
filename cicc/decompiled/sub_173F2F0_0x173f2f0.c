// Function: sub_173F2F0
// Address: 0x173f2f0
//
bool __fastcall sub_173F2F0(__int64 a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // rdx

  result = 0;
  if ( *(_BYTE *)(a2 + 16) == 78 )
  {
    v3 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v3 + 16) && *(_DWORD *)(v3 + 36) == *(_DWORD *)a1 )
      return *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL)
                       + 24
                       * (*(unsigned int *)(a1 + 8)
                        - (unsigned __int64)(*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) == *(_QWORD *)(a1 + 16);
  }
  return result;
}

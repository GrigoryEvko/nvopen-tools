// Function: sub_21F2100
// Address: 0x21f2100
//
__int64 __fastcall sub_21F2100(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rdx
  unsigned int v4; // ebx
  int v5; // eax
  int v6; // edx
  int v7; // eax

  result = sub_15F3040(a1);
  if ( !(_BYTE)result )
    return 0;
  v2 = *(_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) - 24);
  if ( *(_BYTE *)(v2 + 16) == 20 )
    return *(unsigned __int8 *)(v2 + 96);
  v3 = *(_QWORD *)(a1 - 24);
  if ( !*(_BYTE *)(v3 + 16) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
  {
    v4 = *(_DWORD *)(v3 + 36);
    if ( !sub_1C301F0(v4) )
    {
      LOBYTE(v5) = v4 == 4046;
      LOBYTE(v6) = v4 == 4242;
      v7 = v6 | v5;
      LOBYTE(v6) = v4 == 4405;
      return (v6 | v7) ^ 1u;
    }
    return 0;
  }
  return result;
}

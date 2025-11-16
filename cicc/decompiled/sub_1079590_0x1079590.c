// Function: sub_1079590
// Address: 0x1079590
//
__int64 __fastcall sub_1079590(__int64 a1)
{
  unsigned int v1; // r12d
  char v2; // al
  void *v4; // rax

  v1 = 1;
  v2 = *(_BYTE *)(a1 + 9);
  if ( (v2 & 8) == 0 )
  {
    v1 = *(unsigned __int8 *)(a1 + 44);
    if ( !(_BYTE)v1 )
    {
      if ( !*(_BYTE *)(a1 + 42)
        || *(_QWORD *)a1
        || (v2 & 0x70) == 0x20
        && *(char *)(a1 + 8) >= 0
        && (*(_BYTE *)(a1 + 8) |= 8u, v4 = sub_E807D0(*(_QWORD *)(a1 + 24)), (*(_QWORD *)a1 = v4) != 0) )
      {
        if ( (*(_BYTE *)(a1 + 8) & 2) == 0 && (!*(_BYTE *)(a1 + 36) || *(_DWORD *)(a1 + 32) != 3) )
          return *(unsigned __int8 *)(a1 + 43) ^ 1u;
      }
    }
  }
  return v1;
}

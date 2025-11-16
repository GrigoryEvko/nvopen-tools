// Function: sub_2FDE950
// Address: 0x2fde950
//
char __fastcall sub_2FDE950(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rax
  int v4; // eax

  v2 = *(_DWORD *)(a2 + 44);
  if ( (v2 & 4) == 0 && (v2 & 8) != 0 )
  {
    LOBYTE(v3) = sub_2E88A90(a2, 32, 1);
    if ( !(_BYTE)v3 )
      return v3;
  }
  else
  {
    v3 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 5) & 1LL;
    if ( (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) & 0x20LL) == 0 )
      return v3;
  }
  v4 = *(_DWORD *)(a2 + 44);
  if ( (v4 & 4) != 0 || (v4 & 8) == 0 )
    LOBYTE(v3) = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 7;
  else
    LOBYTE(v3) = sub_2E88A90(a2, 128, 1);
  return v3;
}

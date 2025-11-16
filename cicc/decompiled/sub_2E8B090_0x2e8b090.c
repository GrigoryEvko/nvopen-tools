// Function: sub_2E8B090
// Address: 0x2e8b090
//
char __fastcall sub_2E8B090(__int64 a1)
{
  int v1; // eax
  __int64 v2; // rax

  v1 = *(_DWORD *)(a1 + 44);
  if ( (v1 & 4) == 0 && (v1 & 8) != 0 )
    LOBYTE(v2) = sub_2E88A90(a1, (__int64)&loc_1000000, 1);
  else
    v2 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 24) & 1LL;
  if ( !(_BYTE)v2 && (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 <= 1 )
    return *(_QWORD *)(*(_QWORD *)(a1 + 32) + 64LL) & 1LL;
  return v2;
}

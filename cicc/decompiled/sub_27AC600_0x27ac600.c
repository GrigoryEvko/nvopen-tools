// Function: sub_27AC600
// Address: 0x27ac600
//
bool __fastcall sub_27AC600(__int64 *a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rax

  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(a3 - 8);
  else
    v3 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  v4 = *a1;
  v5 = 32LL * a2;
  if ( (*(_BYTE *)(*a1 + 7) & 0x40) != 0 )
    v6 = *(_QWORD *)(v4 - 8);
  else
    v6 = v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
  return *(_QWORD *)(v6 + v5) != *(_QWORD *)(v3 + v5);
}

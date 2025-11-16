// Function: sub_2F42240
// Address: 0x2f42240
//
__int64 __fastcall sub_2F42240(__int64 a1, unsigned int a2, int a3)
{
  __int64 v3; // rcx
  unsigned int v5; // edx
  __int64 result; // rax

  v3 = *(_QWORD *)(a1 + 16);
  v5 = *(_DWORD *)(*(_QWORD *)(v3 + 8) + 24LL * a2 + 16) & 0xFFF;
  result = *(_QWORD *)(v3 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v3 + 8) + 24LL * a2 + 16) >> 12);
  do
  {
    if ( !result )
      break;
    result += 2;
    *(_DWORD *)(*(_QWORD *)(a1 + 808) + 4LL * v5) = a3;
    v5 += *(__int16 *)(result - 2);
  }
  while ( *(_WORD *)(result - 2) );
  return result;
}

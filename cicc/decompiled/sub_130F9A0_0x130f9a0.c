// Function: sub_130F9A0
// Address: 0x130f9a0
//
__int64 __fastcall sub_130F9A0(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // rsi
  unsigned __int64 v3; // rcx
  unsigned int v4; // ecx
  __int64 v5; // rax
  int v7; // eax

  v1 = a1 - *(_QWORD *)&dword_50607C0;
  v2 = a1 - *(_QWORD *)&dword_50607C0 + 1;
  if ( v2 > 0x7000000000000000LL )
  {
    v5 = 198;
    return qword_5060180[v5] + *(_QWORD *)&dword_50607C0;
  }
  _BitScanReverse64(&v3, v2);
  v4 = v3 - (((v1 & v2) == 0) - 1);
  if ( v4 < 0xE )
    v4 = 14;
  if ( v4 != 14 )
  {
    v5 = 4 * (v4 - 14) + ((v1 >> ((unsigned __int8)v4 - 3)) & 3) - 1;
    return qword_5060180[v5] + *(_QWORD *)&dword_50607C0;
  }
  v7 = (v1 >> 12) & 3;
  if ( v7 )
  {
    v5 = (unsigned int)(v7 - 1);
    return qword_5060180[v5] + *(_QWORD *)&dword_50607C0;
  }
  return a1;
}

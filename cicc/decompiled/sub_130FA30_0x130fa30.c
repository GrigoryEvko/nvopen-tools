// Function: sub_130FA30
// Address: 0x130fa30
//
unsigned __int64 __fastcall sub_130FA30(unsigned __int64 a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rcx
  unsigned __int64 v4; // rdx
  unsigned int v5; // edx
  char v6; // cl
  unsigned int v7; // edx
  __int64 v8; // rax

  result = sub_130F9A0(a1);
  if ( a1 > result )
  {
    v2 = result - *(_QWORD *)&dword_50607C0;
    v3 = v2 + 1;
    if ( v2 + 1 > 0x7000000000000000LL )
    {
      v8 = 199;
    }
    else
    {
      _BitScanReverse64(&v4, v3);
      v5 = v4 - (((v2 & v3) == 0) - 1);
      if ( v5 < 0xE )
        v5 = 14;
      v6 = v5 - 3;
      v7 = v5 - 14;
      if ( !v7 )
        v6 = 12;
      v8 = ((v2 >> v6) & 3) + 4 * v7;
    }
    return qword_5060180[v8] + *(_QWORD *)&dword_50607C0;
  }
  return result;
}

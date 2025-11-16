// Function: sub_109DDE0
// Address: 0x109dde0
//
__int64 __fastcall sub_109DDE0(__int64 a1, unsigned int a2, int a3)
{
  unsigned int v4; // r8d
  unsigned int v5; // esi

  *(_DWORD *)(a1 + 8) = a2;
  if ( a2 > 0x40 )
  {
    sub_C43690(a1, 0, 0);
    v4 = *(_DWORD *)(a1 + 8);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    v4 = a2;
  }
  v5 = v4 - a3;
  if ( v4 - a3 == v4 )
    return a1;
  if ( v5 <= 0x3F && v4 <= 0x40 )
  {
    *(_QWORD *)a1 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a3) << v5;
    return a1;
  }
  sub_C43C90((_QWORD *)a1, v5, v4);
  return a1;
}

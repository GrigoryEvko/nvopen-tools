// Function: sub_E95020
// Address: 0xe95020
//
unsigned __int64 __fastcall sub_E95020(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        int a7,
        char a8,
        __int64 a9)
{
  int v9; // r14d
  unsigned __int64 result; // rax
  char v12; // dl

  v9 = a6;
  LOBYTE(a6) = 0;
  if ( (unsigned __int8)v9 <= 0x12u )
    a6 = (0x41002uLL >> v9) & 1;
  sub_E92760(a1, 3, a4, a5, (unsigned __int8)(a8 - 2) <= 1u, a6, a9);
  *(_DWORD *)(a1 + 164) = v9;
  *(_DWORD *)(a1 + 172) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)a1 = &unk_49E3628;
  *(_DWORD *)(a1 + 168) = a7;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  for ( result = 0; result != 16; ++result )
  {
    v12 = 0;
    if ( a3 > result )
      v12 = *(_BYTE *)(a2 + result);
    *(_BYTE *)(a1 + result + 148) = v12;
  }
  return result;
}

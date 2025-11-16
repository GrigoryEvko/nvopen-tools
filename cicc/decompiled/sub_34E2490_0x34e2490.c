// Function: sub_34E2490
// Address: 0x34e2490
//
__int64 __fastcall sub_34E2490(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  if ( a3 != 23
    || (result = *(_QWORD *)a2 ^ 0x6156206775626544LL, (a3 = result | *(_QWORD *)(a2 + 8) ^ 0x4120656C62616972LL) != 0)
    || *(_DWORD *)(a2 + 16) != 2037145966
    || *(_WORD *)(a2 + 20) != 26995
    || *(_BYTE *)(a2 + 22) != 115 )
  {
    sub_31413C0(a1, a2, a3, (__int64)a4, a5, a6);
    return sub_34E2200(a1, a4, 1u);
  }
  return result;
}

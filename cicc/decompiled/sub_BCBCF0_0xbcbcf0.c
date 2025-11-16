// Function: sub_BCBCF0
// Address: 0xbcbcf0
//
__int64 __fastcall sub_BCBCF0(__int64 a1, __int64 a2, int a3)
{
  int v4; // edx
  __int64 result; // rax

  *(_BYTE *)(a1 + 8) = 14;
  v4 = *(unsigned __int8 *)(a1 + 8);
  result = (unsigned int)(a3 << 8);
  *(_QWORD *)a1 = a2;
  *(_DWORD *)(a1 + 12) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 8) = result | v4;
  return result;
}

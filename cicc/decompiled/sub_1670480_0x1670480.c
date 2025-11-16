// Function: sub_1670480
// Address: 0x1670480
//
__int64 __fastcall sub_1670480(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_QWORD *)a1 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 + 8) = *(unsigned int *)(a2 + 12);
  result = (*(_DWORD *)(a2 + 8) >> 9) & 1;
  *(_BYTE *)(a1 + 16) = (*(_DWORD *)(a2 + 8) & 0x200) != 0;
  return result;
}

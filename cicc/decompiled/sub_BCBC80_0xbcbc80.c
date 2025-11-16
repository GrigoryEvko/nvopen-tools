// Function: sub_BCBC80
// Address: 0xbcbc80
//
__int64 __fastcall sub_BCBC80(__int64 a1, __int64 *a2, int a3, unsigned __int8 a4)
{
  __int64 v4; // rax
  __int64 result; // rax

  v4 = *a2;
  *(_QWORD *)(a1 + 24) = a2;
  *(_DWORD *)(a1 + 32) = a3;
  *(_QWORD *)a1 = v4;
  *(_QWORD *)(a1 + 16) = a1 + 24;
  result = a4 | 0x100000000LL;
  *(_QWORD *)(a1 + 8) = result;
  return result;
}

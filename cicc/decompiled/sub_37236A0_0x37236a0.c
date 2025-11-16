// Function: sub_37236A0
// Address: 0x37236a0
//
__int64 __fastcall sub_37236A0(__int64 a1, __int64 a2, int a3, __int16 a4)
{
  __int64 result; // rax

  *(_QWORD *)(a1 + 8) = a2;
  *(_BYTE *)(a1 + 16) = 0;
  *(_BYTE *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A35768;
  result = *(unsigned __int16 *)(a2 + 28);
  *(_WORD *)(a1 + 42) = a4 << 15;
  *(_WORD *)(a1 + 40) = result;
  *(_DWORD *)(a1 + 44) = a3;
  return result;
}

// Function: sub_23A0D00
// Address: 0x23a0d00
//
__int64 __fastcall sub_23A0D00(__int64 a1)
{
  char v1; // al
  __int64 result; // rax

  v1 = byte_4FDD768;
  *(_DWORD *)a1 = 16777473;
  *(_DWORD *)(a1 + 20) = -1;
  *(_BYTE *)(a1 + 4) = v1;
  *(_BYTE *)(a1 + 5) = qword_5003700[17];
  *(_DWORD *)(a1 + 8) = qword_4FFDE88[8];
  *(_DWORD *)(a1 + 12) = qword_4FFDDA8[8];
  *(_WORD *)(a1 + 16) = 1;
  *(_BYTE *)(a1 + 18) = byte_4FDDCA8;
  result = (unsigned __int8)byte_4FDDD88;
  *(_BYTE *)(a1 + 24) = byte_4FDDD88;
  return result;
}

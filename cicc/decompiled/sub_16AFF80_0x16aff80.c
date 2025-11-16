// Function: sub_16AFF80
// Address: 0x16aff80
//
__int64 __fastcall sub_16AFF80(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 160);
  *(_BYTE *)(result + 8) = *(_BYTE *)(a1 + 192);
  return result;
}

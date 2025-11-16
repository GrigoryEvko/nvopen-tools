// Function: sub_7E99A0
// Address: 0x7e99a0
//
__int64 __fastcall sub_7E99A0(__int64 a1)
{
  __int64 result; // rax

  sub_80CF50();
  result = *(_DWORD *)(a1 + 88) & 0xFFF7FF8F | (16 * (unsigned __int8)(((*(_BYTE *)(a1 - 8) & 8) != 0) + 2));
  *(_DWORD *)(a1 + 88) = result;
  return result;
}

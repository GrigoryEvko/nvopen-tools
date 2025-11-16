// Function: sub_15CE2C0
// Address: 0x15ce2c0
//
__int64 __fastcall sub_15CE2C0(__int64 a1)
{
  __int64 result; // rax

  result = sub_15CE0F0(a1 + 184);
  *(_BYTE *)(a1 + 232) = 0;
  *(_DWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_DWORD *)(a1 + 236) = 0;
  return result;
}

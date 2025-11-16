// Function: sub_641570
// Address: 0x641570
//
__int64 __fastcall sub_641570(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 126);
  if ( (result & 0x20) != 0 )
  {
    --*(_BYTE *)(qword_4F061C8 + 83LL);
    result = *(_BYTE *)(a1 + 126) & 0xDF;
    *(_BYTE *)(a1 + 126) &= ~0x20u;
  }
  if ( (result & 0x40) != 0 )
  {
    --*(_BYTE *)(qword_4F061C8 + 75LL);
    result = *(_BYTE *)(a1 + 126) & 0xBF;
    *(_BYTE *)(a1 + 126) &= ~0x40u;
  }
  if ( (result & 0x80u) != 0LL )
  {
    result = qword_4F061C8;
    --*(_BYTE *)(qword_4F061C8 + 64LL);
    *(_BYTE *)(a1 + 126) &= ~0x80u;
  }
  if ( (*(_BYTE *)(a1 + 127) & 1) != 0 )
  {
    result = qword_4F061C8;
    --*(_BYTE *)(qword_4F061C8 + 81LL);
    *(_BYTE *)(a1 + 127) &= ~1u;
  }
  return result;
}

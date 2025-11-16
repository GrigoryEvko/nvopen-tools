// Function: sub_8D10F0
// Address: 0x8d10f0
//
__int64 __fastcall sub_8D10F0(__int64 a1, char a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = (unsigned int)*(unsigned __int8 *)(a1 + 140) - 9;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
  {
    v3 = *(_QWORD *)(a1 + 168);
    result = *(_WORD *)(v3 + 111) & 0xFE7F;
    *(_WORD *)(v3 + 111) = result | ((a2 & 1) << 7) | 0x100;
  }
  return result;
}

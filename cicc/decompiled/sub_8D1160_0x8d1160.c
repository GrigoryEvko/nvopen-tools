// Function: sub_8D1160
// Address: 0x8d1160
//
__int64 __fastcall sub_8D1160(__int64 a1, char a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = (unsigned int)*(unsigned __int8 *)(a1 + 140) - 9;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
  {
    v3 = *(_QWORD *)(a1 + 168);
    result = *(_BYTE *)(v3 + 111) & 0xE7;
    *(_BYTE *)(v3 + 111) = *(_BYTE *)(v3 + 111) & 0xE7 | (8 * (a2 & 1)) | 0x10;
  }
  return result;
}

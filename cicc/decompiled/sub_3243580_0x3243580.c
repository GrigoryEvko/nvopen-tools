// Function: sub_3243580
// Address: 0x3243580
//
__int64 __fastcall sub_3243580(__int64 a1, _BYTE *a2)
{
  unsigned int v2; // eax
  unsigned __int16 v3; // dx
  __int16 v4; // dx
  __int64 result; // rax

  v2 = *(unsigned __int16 *)(a1 + 100);
  v3 = v2;
  LOWORD(v2) = v2 & 0xFE3F;
  v4 = (v3 >> 6) & 7;
  result = ((v4 & 6 | 1) << 6) | v2;
  *(_WORD *)(a1 + 100) = result;
  if ( !*a2 )
  {
    LOWORD(result) = result & 0xFE3F;
    result = ((v4 & 4 | 3) << 6) | (unsigned int)result;
    *(_WORD *)(a1 + 100) = result;
  }
  return result;
}

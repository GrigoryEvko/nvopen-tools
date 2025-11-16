// Function: sub_B2F600
// Address: 0xb2f600
//
__int64 __fastcall sub_B2F600(__int64 a1)
{
  __int64 result; // rax
  unsigned int v2; // eax

  result = 0;
  if ( !*(_BYTE *)a1 )
  {
    v2 = *(unsigned __int16 *)(a1 + 34);
    LOWORD(v2) = (unsigned __int16)v2 >> 1;
    return (v2 >> 10) & 1;
  }
  return result;
}

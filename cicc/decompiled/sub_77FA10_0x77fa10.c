// Function: sub_77FA10
// Address: 0x77fa10
//
__int64 __fastcall sub_77FA10(int a1, __int64 a2)
{
  __int64 result; // rax

  result = (unsigned int)dword_4F08058;
  if ( dword_4F08058 )
  {
    result = sub_771BE0();
    dword_4F08058 = 0;
  }
  *(_DWORD *)(a2 + 224) = a1;
  *(_BYTE *)(a2 + 193) |= 8u;
  return result;
}

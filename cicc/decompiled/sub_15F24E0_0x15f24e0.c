// Function: sub_15F24E0
// Address: 0x15f24e0
//
__int64 __fastcall sub_15F24E0(__int64 a1)
{
  __int64 result; // rax

  result = *(_BYTE *)(a1 + 17) >> 1;
  if ( (_DWORD)result == 127 )
    return 0xFFFFFFFFLL;
  return result;
}

// Function: sub_2EC1A40
// Address: 0x2ec1a40
//
unsigned __int64 __fastcall sub_2EC1A40(unsigned __int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v3; // rdx
  __int16 v4; // dx

  result = a1;
  do
  {
    result = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
    if ( !result )
      BUG();
    v3 = *(_QWORD *)result;
    if ( (*(_QWORD *)result & 4) == 0 && (*(_BYTE *)(result + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        result = v3 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)((v3 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
          break;
        v3 = *(_QWORD *)result;
      }
    }
    if ( a2 == result )
      break;
    v4 = *(_WORD *)(result + 68);
  }
  while ( (unsigned __int16)(v4 - 14) <= 4u || v4 == 24 );
  return result;
}

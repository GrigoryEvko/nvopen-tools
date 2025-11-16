// Function: sub_1E6C1C0
// Address: 0x1e6c1c0
//
unsigned __int64 __fastcall sub_1E6C1C0(unsigned __int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v3; // rdx

  result = a1;
  do
  {
    result = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
    if ( !result )
      BUG();
    v3 = *(_QWORD *)result;
    if ( (*(_QWORD *)result & 4) == 0 && (*(_BYTE *)(result + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        result = v3 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)((v3 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
          break;
        v3 = *(_QWORD *)result;
      }
    }
  }
  while ( a2 != result && (unsigned __int16)(**(_WORD **)(result + 16) - 12) <= 1u );
  return result;
}

// Function: sub_34C10E0
// Address: 0x34c10e0
//
unsigned __int64 __fastcall sub_34C10E0(unsigned __int64 *a1)
{
  unsigned __int64 result; // rax
  __int64 v2; // rdx

  result = *(_QWORD *)*a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !result )
    BUG();
  v2 = *(_QWORD *)result;
  if ( (*(_QWORD *)result & 4) == 0 && (*(_BYTE *)(result + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      result = v2 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)((v2 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
        break;
      v2 = *(_QWORD *)result;
    }
  }
  *a1 = result;
  return result;
}

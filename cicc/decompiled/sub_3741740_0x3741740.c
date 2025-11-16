// Function: sub_3741740
// Address: 0x3741740
//
unsigned __int64 __fastcall sub_3741740(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  unsigned __int64 result; // rax
  __int64 v4; // rdx

  v2 = *(_QWORD *)(a1 + 40);
  result = *(_QWORD *)(v2 + 752);
  if ( result != *(_QWORD *)(*(_QWORD *)(v2 + 744) + 56LL) )
  {
    result = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
    if ( !result )
      BUG();
    v4 = *(_QWORD *)result;
    if ( (*(_QWORD *)result & 4) == 0 && (*(_BYTE *)(result + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        result = v4 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
          break;
        v4 = *(_QWORD *)result;
      }
    }
    *(_QWORD *)(a1 + 160) = result;
  }
  *(_QWORD *)(v2 + 752) = a2;
  return result;
}

// Function: sub_3741010
// Address: 0x3741010
//
unsigned __int64 __fastcall sub_3741010(_QWORD *a1)
{
  __int64 v1; // rcx
  __int64 *v2; // rdx
  unsigned __int64 result; // rax
  __int64 v4; // rdx

  a1[21] = 0;
  v1 = *(_QWORD *)(a1[5] + 744LL);
  v2 = (__int64 *)(*(_QWORD *)(v1 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  result = (unsigned __int64)v2;
  if ( v2 == (__int64 *)(v1 + 48) )
  {
    a1[20] = 0;
    return 0;
  }
  else
  {
    if ( !v2 )
      BUG();
    v4 = *v2;
    if ( (v4 & 4) == 0 && (*(_BYTE *)(result + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        result = v4 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
          break;
        v4 = *(_QWORD *)result;
      }
    }
    a1[21] = result;
    a1[20] = result;
  }
  return result;
}

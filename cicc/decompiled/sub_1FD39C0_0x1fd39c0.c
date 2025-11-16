// Function: sub_1FD39C0
// Address: 0x1fd39c0
//
unsigned __int64 __fastcall sub_1FD39C0(_QWORD *a1)
{
  __int64 v1; // rcx
  __int64 *v2; // rdx
  unsigned __int64 result; // rax
  __int64 v4; // rdx

  a1[19] = 0;
  v1 = *(_QWORD *)(a1[5] + 784LL);
  v2 = (__int64 *)(*(_QWORD *)(v1 + 24) & 0xFFFFFFFFFFFFFFF8LL);
  result = (unsigned __int64)v2;
  if ( v2 == (__int64 *)(v1 + 24) )
  {
    a1[18] = 0;
    return 0;
  }
  else
  {
    if ( !v2 )
      BUG();
    v4 = *v2;
    if ( (v4 & 4) == 0 && (*(_BYTE *)(result + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        result = v4 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
          break;
        v4 = *(_QWORD *)result;
      }
    }
    a1[19] = result;
    a1[18] = result;
  }
  return result;
}

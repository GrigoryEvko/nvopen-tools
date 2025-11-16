// Function: sub_1E31310
// Address: 0x1e31310
//
__int64 __fastcall sub_1E31310(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // rax

  result = 0;
  if ( *(char *)(a1 + 3) < 0 )
  {
    v2 = *(_QWORD *)(a1 + 16);
    if ( v2 )
    {
      v3 = *(_QWORD *)(*(_QWORD *)(v2 + 16) + 8LL);
      if ( (*(_BYTE *)(a1 + 3) & 0x10) != 0 )
        return (v3 & 0x10000000) == 0;
      else
        return (v3 & 0x8000000) == 0;
    }
    else
    {
      return 1;
    }
  }
  return result;
}

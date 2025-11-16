// Function: sub_2EAB300
// Address: 0x2eab300
//
__int64 __fastcall sub_2EAB300(__int64 a1)
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
      v3 = *(_QWORD *)(*(_QWORD *)(v2 + 16) + 24LL);
      if ( (*(_BYTE *)(a1 + 3) & 0x10) != 0 )
        return !(v3 & 0x100000000LL);
      else
        return (int)v3 >= 0;
    }
    else
    {
      return 1;
    }
  }
  return result;
}

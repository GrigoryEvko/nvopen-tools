// Function: sub_71ACC0
// Address: 0x71acc0
//
__int64 __fastcall sub_71ACC0(__int64 a1)
{
  __int64 result; // rax
  char v2; // dl

  result = 1;
  if ( *(_BYTE *)(a1 + 173) == 6 )
  {
    v2 = *(_BYTE *)(a1 + 176);
    if ( v2 == 1 )
    {
      return 1 - (*(_BYTE *)(*(_QWORD *)(a1 + 184) + 89LL) & 1u);
    }
    else if ( v2 == 3 )
    {
      return *(_BYTE *)(*(_QWORD *)(a1 + 184) - 8LL) & 1;
    }
  }
  return result;
}

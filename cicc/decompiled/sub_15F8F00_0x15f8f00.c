// Function: sub_15F8F00
// Address: 0x15f8f00
//
__int64 __fastcall sub_15F8F00(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rcx
  __int64 v3; // rdx

  result = 0;
  if ( *(_BYTE *)(*(_QWORD *)(a1 - 24) + 16LL) == 13 )
  {
    v2 = *(_QWORD *)(a1 + 40);
    v3 = *(_QWORD *)(*(_QWORD *)(v2 + 56) + 80LL);
    if ( v3 )
    {
      if ( v2 == v3 - 24 )
        return ((unsigned __int8)(*(_WORD *)(a1 + 18) >> 5) ^ 1) & 1;
    }
  }
  return result;
}

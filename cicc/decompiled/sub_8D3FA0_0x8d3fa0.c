// Function: sub_8D3FA0
// Address: 0x8d3fa0
//
__int64 __fastcall sub_8D3FA0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // rdi

  result = a1;
  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 32LL);
  if ( *(_BYTE *)(v2 + 80) == 19 )
  {
    v3 = *(_QWORD *)(v2 + 88);
    if ( (*(_BYTE *)(v3 + 266) & 1) != 0 )
    {
      v4 = *(_QWORD *)(v3 + 200);
      if ( v2 != v4 )
        return sub_72EF10(v4);
    }
  }
  return result;
}

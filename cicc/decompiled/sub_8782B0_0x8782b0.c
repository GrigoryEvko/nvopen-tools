// Function: sub_8782B0
// Address: 0x8782b0
//
__int64 __fastcall sub_8782B0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = 0;
  if ( *(_BYTE *)(a1 + 80) == 2 )
  {
    v2 = *(_QWORD *)(a1 + 88);
    if ( v2 )
    {
      if ( *(_BYTE *)(v2 + 173) == 12 )
        return (*(_BYTE *)(v2 + 176) == 2) | (unsigned __int8)(*(_BYTE *)(v2 + 176) == 13);
    }
  }
  return result;
}

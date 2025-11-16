// Function: sub_6E9790
// Address: 0x6e9790
//
__int64 __fastcall sub_6E9790(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  *a2 = 0;
  result = 0;
  if ( *(_BYTE *)(a1 + 16) == 1 )
  {
    v3 = *(_QWORD *)(a1 + 144);
    if ( *(_BYTE *)(v3 + 24) == 3 && (*(_BYTE *)(v3 + 25) & 1) != 0 )
    {
      *a2 = *(_QWORD *)(v3 + 56);
      return 1;
    }
  }
  return result;
}

// Function: sub_223F1E0
// Address: 0x223f1e0
//
__int64 __fastcall sub_223F1E0(__int64 a1)
{
  unsigned __int64 v1; // rdx
  unsigned __int64 v2; // rax

  if ( (*(_BYTE *)(a1 + 64) & 8) == 0 )
    return -1;
  v1 = *(_QWORD *)(a1 + 40);
  v2 = *(_QWORD *)(a1 + 24);
  if ( v1 )
  {
    if ( v1 > v2 )
    {
      *(_QWORD *)(a1 + 24) = v1;
      v2 = v1;
    }
  }
  return v2 - *(_QWORD *)(a1 + 16);
}

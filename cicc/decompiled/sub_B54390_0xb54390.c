// Function: sub_B54390
// Address: 0xb54390
//
__int64 __fastcall sub_B54390(__int64 a1)
{
  bool v1; // zf

  v1 = *(_BYTE *)(a1 + 56) == 0;
  *(_BYTE *)(a1 + 64) = 0;
  if ( !v1 && *(_DWORD *)(a1 + 16) )
    *(_DWORD *)(a1 + 16) = 0;
  return sub_B43D60(*(_QWORD **)a1);
}

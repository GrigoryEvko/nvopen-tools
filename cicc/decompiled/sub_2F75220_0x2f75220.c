// Function: sub_2F75220
// Address: 0x2f75220
//
void __fastcall sub_2F75220(__int64 a1, __int64 a2)
{
  if ( *(_QWORD *)(a1 + 448) == a2 )
  {
    *(_QWORD *)(a1 + 448) = 0;
    *(_DWORD *)(a1 + 32) = 0;
  }
}

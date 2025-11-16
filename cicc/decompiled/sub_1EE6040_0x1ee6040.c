// Function: sub_1EE6040
// Address: 0x1ee6040
//
void __fastcall sub_1EE6040(__int64 a1, __int64 a2)
{
  if ( *(_QWORD *)(a1 + 192) == a2 )
  {
    *(_QWORD *)(a1 + 192) = 0;
    *(_DWORD *)(a1 + 32) = 0;
  }
}

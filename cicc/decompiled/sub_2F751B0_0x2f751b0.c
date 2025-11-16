// Function: sub_2F751B0
// Address: 0x2f751b0
//
void __fastcall sub_2F751B0(__int64 a1, __int64 a2)
{
  if ( *(_QWORD *)(a1 + 440) == a2 )
  {
    *(_QWORD *)(a1 + 440) = 0;
    *(_DWORD *)(a1 + 32) = 0;
  }
}

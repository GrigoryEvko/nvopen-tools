// Function: sub_7E0550
// Address: 0x7e0550
//
void __fastcall sub_7E0550(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  if ( *(_BYTE *)(a1 + 24) == 5 )
  {
    v2 = *(_QWORD *)(a1 + 56);
    if ( *(_QWORD *)(v2 + 16) )
    {
      if ( *(_QWORD *)(v2 + 24) )
      {
        *(_DWORD *)(a2 + 80) = 1;
        *(_DWORD *)(a2 + 72) = 1;
      }
    }
  }
}

// Function: sub_728FD0
// Address: 0x728fd0
//
void __fastcall sub_728FD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  if ( *(_BYTE *)(a1 + 48) == 5 )
  {
    v2 = *(_QWORD *)(a1 + 56);
    if ( v2 )
    {
      if ( (*(_BYTE *)(v2 + 194) & 6) == 0 )
      {
        *(_DWORD *)(a2 + 80) = 1;
        *(_DWORD *)(a2 + 72) = 1;
      }
    }
  }
}

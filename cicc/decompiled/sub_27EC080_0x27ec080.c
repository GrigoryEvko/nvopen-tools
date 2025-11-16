// Function: sub_27EC080
// Address: 0x27ec080
//
bool __fastcall sub_27EC080(__int64 a1)
{
  __int64 v1; // rax
  __int64 i; // rcx
  __int64 v3; // rdx

  v1 = *(_QWORD *)(a1 + 56);
  for ( i = a1 + 48; i != v1; v1 = *(_QWORD *)(v1 + 8) )
  {
    if ( !v1 )
      BUG();
    if ( *(_BYTE *)(v1 - 24) == 85 )
    {
      v3 = *(_QWORD *)(v1 - 56);
      if ( v3 )
      {
        if ( !*(_BYTE *)v3
          && *(_QWORD *)(v3 + 24) == *(_QWORD *)(v1 + 56)
          && (*(_BYTE *)(v3 + 33) & 0x20) != 0
          && *(_DWORD *)(v3 + 36) == 60 )
        {
          break;
        }
      }
    }
  }
  return i != v1;
}

// Function: sub_728F70
// Address: 0x728f70
//
void __fastcall sub_728F70(__int64 a1)
{
  __int64 v1; // rax
  char v2; // al

  if ( a1 )
  {
    if ( dword_4F04C44 == -1 )
    {
      v1 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v1 + 6) & 6) == 0 && *(_BYTE *)(v1 + 4) != 12 )
      {
        v2 = *(_BYTE *)(a1 + 80);
        if ( (unsigned __int8)(v2 - 9) <= 2u || v2 == 7 )
          sub_8AD0D0(a1, 1, 0);
      }
    }
  }
}

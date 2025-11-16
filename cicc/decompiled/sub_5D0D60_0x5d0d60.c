// Function: sub_5D0D60
// Address: 0x5d0d60
//
void __fastcall sub_5D0D60(_BYTE *a1, int a2)
{
  __int64 v2; // rax

  if ( !*a1 )
  {
    v2 = unk_4F04C68 + 776LL * unk_4F04C64;
    if ( *(_BYTE *)(v2 + 4) == 6 && a2 )
    {
      *a1 = *(_BYTE *)(v2 + 704);
    }
    else if ( unk_4F04C34 != -1 && unk_4F04C58 == -1 )
    {
      if ( qword_4CF6E40 )
        *a1 = *(_BYTE *)(qword_4CF6E40 + 8);
    }
  }
}

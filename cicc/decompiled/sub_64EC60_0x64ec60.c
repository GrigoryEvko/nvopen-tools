// Function: sub_64EC60
// Address: 0x64ec60
//
void __fastcall sub_64EC60(__int64 a1)
{
  __int64 v1; // rax

  if ( (*(_BYTE *)(a1 + 8) & 0x20) != 0 && dword_4F077C4 == 2 )
  {
    if ( !dword_4F077BC
      || qword_4F077A8 > 0x76BFu
      || unk_4D0448C
      || *(char *)(a1 + 122) < 0
      || (*(_BYTE *)(a1 + 123) & 1) != 0
      && (dword_4F04C44 != -1
       || (v1 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v1 + 6) & 6) != 0)
       || *(_BYTE *)(v1 + 4) == 12) )
    {
      sub_6851C0(255, a1 + 24);
      *(_QWORD *)(a1 + 288) = sub_72C930(255);
    }
  }
}

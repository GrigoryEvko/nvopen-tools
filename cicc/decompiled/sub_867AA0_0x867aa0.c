// Function: sub_867AA0
// Address: 0x867aa0
//
__int64 sub_867AA0()
{
  unsigned int v0; // r8d
  __int64 v1; // rax

  v0 = 0;
  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 4) != 0 )
  {
    v1 = *(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 208);
    if ( v1 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(v1 + 140) - 9) <= 2u )
        return (*(_BYTE *)(v1 + 177) & 0x40) != 0;
    }
  }
  return v0;
}

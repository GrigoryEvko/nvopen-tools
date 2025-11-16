// Function: sub_861C90
// Address: 0x861c90
//
__int64 sub_861C90()
{
  __int64 result; // rax
  __int64 v1; // rcx
  char v2; // si
  __int64 v3; // rax
  __int64 v4; // rdx

  result = unk_4F04C48;
  if ( unk_4F04C48 == -1 )
    return sub_6851C0(0xB08u, dword_4F07508);
  v1 = qword_4F04C68[0];
  v2 = *(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 12);
  if ( v2 >= 0 )
  {
    while ( 1 )
    {
      v3 = 776 * result;
      v4 = v1 + v3;
      if ( (v2 & 0x10) == 0 )
        break;
      *(_BYTE *)(v4 + 12) |= 0x80u;
      *(_BYTE *)(v4 + 8) &= ~0x20u;
      result = *(int *)(v1 + v3 - 424);
      if ( (_DWORD)result == -1 )
        break;
      v2 = *(_BYTE *)(v1 + 776 * result + 12);
    }
    return sub_6851C0(0xB08u, dword_4F07508);
  }
  return result;
}

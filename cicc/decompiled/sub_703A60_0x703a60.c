// Function: sub_703A60
// Address: 0x703a60
//
__int64 sub_703A60()
{
  unsigned int v0; // r8d
  __int64 v1; // rax

  v0 = 0;
  if ( dword_4F04C58 != -1 )
  {
    v1 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
    if ( v1 )
    {
      if ( (*(_BYTE *)(v1 + 198) & 0x10) != 0 )
        return ((*(_BYTE *)(v1 - 8) >> 4) ^ 1) & 1;
    }
  }
  return v0;
}

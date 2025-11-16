// Function: sub_7E0E30
// Address: 0x7e0e30
//
__int64 sub_7E0E30()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = 0;
  if ( !unk_4D045C4 )
  {
    result = (unsigned int)dword_4F189E8;
    if ( dword_4F189E8 )
      return 1;
    if ( qword_4F04C50 )
    {
      v1 = *(_QWORD *)(qword_4F04C50 + 32LL);
      if ( v1 )
      {
        if ( (*(_BYTE *)(v1 + 198) & 0x10) != 0 )
          return 1;
        result = (unsigned int)dword_4D04530;
        if ( dword_4D04530 )
          return (*(_BYTE *)(v1 + 193) & 2) != 0;
      }
    }
  }
  return result;
}

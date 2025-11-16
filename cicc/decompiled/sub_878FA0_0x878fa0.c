// Function: sub_878FA0
// Address: 0x878fa0
//
__int64 __fastcall sub_878FA0(_QWORD *a1)
{
  __int64 result; // rax

  if ( (*((_BYTE *)a1 + 81) & 0x10) != 0 )
  {
    if ( dword_4F04C44 != -1 || (result = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(result + 6) & 2) != 0) )
    {
      result = a1[8];
      if ( *(char *)(result + 177) < 0 )
        return sub_7D04A0(a1);
    }
  }
  return result;
}

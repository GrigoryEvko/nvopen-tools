// Function: sub_1F44230
// Address: 0x1f44230
//
__int64 __fastcall sub_1F44230(__int64 a1, char a2)
{
  unsigned int *v2; // rax
  __int64 result; // rax

  v2 = (unsigned int *)sub_16D40F0((__int64)qword_4FBB330);
  if ( v2 )
  {
    result = *v2;
    if ( a2 )
      return (unsigned int)dword_4FCB440;
  }
  else
  {
    result = LODWORD(qword_4FBB330[2]);
    if ( a2 )
      return (unsigned int)dword_4FCB440;
  }
  return result;
}

// Function: sub_64A410
// Address: 0x64a410
//
unsigned int *__fastcall sub_64A410(__int64 a1)
{
  unsigned int *result; // rax

  if ( (*(_BYTE *)(a1 + 64) & 2) == 0 )
  {
    result = &dword_4F077BC;
    if ( !dword_4F077BC )
    {
      result = (unsigned int *)*(unsigned int *)(a1 + 24);
      if ( (_DWORD)result )
        return (unsigned int *)sub_684B00(540, a1 + 24);
    }
  }
  return result;
}

// Function: sub_1404680
// Address: 0x1404680
//
void (*__fastcall sub_1404680(__int64 a1, __int64 a2))()
{
  void (*result)(); // rax
  unsigned int v4; // ebx
  __int64 v5; // rdi

  result = (void (*)())*(unsigned int *)(a1 + 192);
  if ( (_DWORD)result )
  {
    v4 = 0;
    do
    {
      while ( 1 )
      {
        v5 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL * v4);
        result = *(void (**)())(*(_QWORD *)v5 + 184LL);
        if ( result != nullsub_522 )
          break;
        if ( *(_DWORD *)(a1 + 192) <= ++v4 )
          return result;
      }
      ++v4;
      result = (void (*)())((__int64 (__fastcall *)(__int64, __int64))result)(v5, a2);
    }
    while ( *(_DWORD *)(a1 + 192) > v4 );
  }
  return result;
}

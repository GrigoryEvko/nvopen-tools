// Function: sub_1404520
// Address: 0x1404520
//
void (*__fastcall sub_1404520(__int64 a1, __int64 a2, __int64 a3, __int64 a4))()
{
  void (*result)(); // rax
  unsigned int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // [rsp-40h] [rbp-40h]

  result = (void (*)())*(unsigned int *)(a1 + 192);
  if ( (_DWORD)result )
  {
    v7 = 0;
    do
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL * v7);
        result = *(void (**)())(*(_QWORD *)v8 + 168LL);
        if ( result != nullsub_520 )
          break;
        if ( *(_DWORD *)(a1 + 192) <= ++v7 )
          return result;
      }
      v9 = a4;
      ++v7;
      result = (void (*)())((__int64 (__fastcall *)(__int64, __int64, __int64))result)(v8, a2, a3);
      a4 = v9;
    }
    while ( *(_DWORD *)(a1 + 192) > v7 );
  }
  return result;
}

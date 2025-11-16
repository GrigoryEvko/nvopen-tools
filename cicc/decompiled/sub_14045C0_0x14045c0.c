// Function: sub_14045C0
// Address: 0x14045c0
//
void (*__fastcall sub_14045C0(__int64 a1, __int64 a2, __int64 a3))()
{
  __int64 v6; // r15
  __int64 i; // rbx
  __int64 v8; // rsi
  void (*result)(); // rax
  unsigned int v10; // ebx
  __int64 v11; // rdi

  if ( *(_BYTE *)(a2 + 16) == 18 )
  {
    v6 = *(_QWORD *)(a2 + 48);
    for ( i = a2 + 40; i != v6; v6 = *(_QWORD *)(v6 + 8) )
    {
      v8 = v6 - 24;
      if ( !v6 )
        v8 = 0;
      sub_14045C0(a1, v8, a3);
    }
  }
  result = (void (*)())*(unsigned int *)(a1 + 192);
  if ( (_DWORD)result )
  {
    v10 = 0;
    do
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL * v10);
        result = *(void (**)())(*(_QWORD *)v11 + 176LL);
        if ( result != nullsub_521 )
          break;
        if ( ++v10 >= *(_DWORD *)(a1 + 192) )
          return result;
      }
      ++v10;
      result = (void (*)())((__int64 (__fastcall *)(__int64, __int64, __int64))result)(v11, a2, a3);
    }
    while ( v10 < *(_DWORD *)(a1 + 192) );
  }
  return result;
}

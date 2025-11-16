// Function: sub_1BF8240
// Address: 0x1bf8240
//
char __fastcall sub_1BF8240(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  unsigned __int16 v3; // cx
  char result; // al
  __int64 v5; // rax
  _QWORD *v6; // r14
  unsigned int v7; // ebx
  __int64 v8; // r13

  while ( 1 )
  {
    v3 = *(_WORD *)(a1 + 24);
    if ( (unsigned __int16)(v3 - 1) > 2u )
      break;
    a1 = *(_QWORD *)(a1 + 32);
  }
  result = (unsigned int)v3 - 4 <= 1 || (unsigned __int16)(v3 - 7) <= 2u;
  if ( result )
  {
    if ( v3 == 7 )
    {
      result = 0;
      if ( *(_QWORD *)(a1 + 48) == a2 )
      {
        result = a3;
        if ( !a3 )
          return (*(_BYTE *)(a1 + 26) & 5) != 0;
      }
    }
    else if ( v3 == 4 && (v5 = *(_QWORD *)(a1 + 40), (_DWORD)v5) )
    {
      v6 = *(_QWORD **)(a1 + 32);
      v7 = a3;
      v8 = (__int64)&v6[(unsigned int)(v5 - 1) + 1];
      do
      {
        result = sub_1BF8240(*v6, a2, v7);
        if ( result )
          break;
        ++v6;
      }
      while ( v6 != (_QWORD *)v8 );
    }
    else
    {
      return 0;
    }
  }
  return result;
}

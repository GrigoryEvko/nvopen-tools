// Function: sub_98CE60
// Address: 0x98ce60
//
__int64 __fastcall sub_98CE60(
        int a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        __int64 a9)
{
  __int64 i; // rbx
  __int64 v11; // rax

  for ( i = a7; a9 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 24) == 85
      && (v11 = *(_QWORD *)(i - 56)) != 0
      && !*(_BYTE *)v11
      && *(_QWORD *)(v11 + 24) == *(_QWORD *)(i + 56)
      && (*(_BYTE *)(v11 + 33) & 0x20) != 0 )
    {
      if ( (unsigned int)(*(_DWORD *)(v11 + 36) - 68) <= 3 )
        continue;
      if ( !--a1 )
        return 0;
    }
    else if ( !--a1 )
    {
      return 0;
    }
    if ( !(unsigned __int8)sub_98CD80((char *)(i - 24)) )
      return 0;
  }
  return 1;
}

// Function: sub_89A020
// Address: 0x89a020
//
__int64 sub_89A020()
{
  __int64 i; // rbx
  __int64 result; // rax

  for ( i = qword_4F601F0; i; i = *(_QWORD *)(i + 8) )
  {
    while ( 1 )
    {
      if ( !*(_QWORD *)(i + 16) )
      {
        result = sub_892270((_QWORD *)i);
        if ( (*(_BYTE *)(i + 80) & 1) != 0 )
          break;
      }
      i = *(_QWORD *)(i + 8);
      if ( !i )
        return result;
    }
    result = *(_QWORD *)(i + 16);
    ++*(_DWORD *)(result + 24);
  }
  return result;
}

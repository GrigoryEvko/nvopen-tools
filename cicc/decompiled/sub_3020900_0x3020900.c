// Function: sub_3020900
// Address: 0x3020900
//
__int64 __fastcall sub_3020900(__int64 a1)
{
  const char *v1; // rcx
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rbx

  if ( !a1 )
    return 0;
  if ( *(_BYTE *)a1 == 3 )
  {
    v1 = sub_BD5D20(a1);
    result = 1;
    if ( v3 == 9 )
    {
      if ( *(_QWORD *)v1 != 0x6573752E6D766C6CLL )
        return 1;
      result = 0;
      if ( v1[8] != 100 )
        return 1;
    }
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 16);
    if ( v4 )
    {
      while ( 1 )
      {
        if ( **(_BYTE **)(v4 + 24) <= 0x15u )
        {
          result = sub_3020900();
          if ( (_BYTE)result )
            break;
        }
        v4 = *(_QWORD *)(v4 + 8);
        if ( !v4 )
          return 0;
      }
    }
    else
    {
      return 0;
    }
  }
  return result;
}

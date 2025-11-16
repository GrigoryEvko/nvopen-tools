// Function: sub_214AC00
// Address: 0x214ac00
//
__int64 __fastcall sub_214AC00(__int64 a1)
{
  const char *v1; // rcx
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rbx
  _QWORD *v5; // rdi

  if ( !a1 )
    return 0;
  if ( *(_BYTE *)(a1 + 16) == 3 )
  {
    v1 = sub_1649960(a1);
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
    v4 = *(_QWORD *)(a1 + 8);
    if ( v4 )
    {
      while ( 1 )
      {
        v5 = sub_1648700(v4);
        if ( *((_BYTE *)v5 + 16) <= 0x10u )
        {
          result = sub_214AC00(v5);
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

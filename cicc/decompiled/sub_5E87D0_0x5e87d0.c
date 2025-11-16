// Function: sub_5E87D0
// Address: 0x5e87d0
//
__int64 __fastcall sub_5E87D0(__int64 a1)
{
  __int64 i; // rax
  __int64 result; // rax
  __int64 *v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rdi

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = *(_QWORD *)i;
  v4 = *(__int64 **)(result + 96);
  v5 = *v4;
  if ( *v4 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        if ( *(_BYTE *)(v5 + 80) == 8 )
        {
          v6 = *(_QWORD *)(v5 + 88);
          if ( (*(_BYTE *)(v6 + 145) & 0x20) != 0 && !*(_QWORD *)(v6 + 152) )
            break;
        }
        v5 = *(_QWORD *)(v5 + 16);
        if ( !v5 )
          return result;
      }
      if ( (*((_BYTE *)v4 + 183) & 0x20) == 0 )
        break;
      result = sub_895AB0();
      v5 = *(_QWORD *)(v5 + 16);
      if ( !v5 )
        return result;
    }
    if ( (*(_BYTE *)(a1 + 141) & 0x20) == 0 )
      return sub_5E8530((_QWORD *)a1, (*(_DWORD *)(a1 + 176) & 0x11000) == 4096);
  }
  return result;
}

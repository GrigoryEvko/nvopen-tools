// Function: sub_691630
// Address: 0x691630
//
__int64 __fastcall sub_691630(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rdx

  v2 = a1;
  result = sub_8D3A70(a1);
  if ( (_DWORD)result )
  {
    if ( *(_BYTE *)(a1 + 140) == 12 )
    {
      v4 = a1;
      do
        v4 = *(_QWORD *)(v4 + 160);
      while ( *(_BYTE *)(v4 + 140) == 12 );
      if ( !*(_QWORD *)v4 )
        return 0;
      do
        v2 = *(_QWORD *)(v2 + 160);
      while ( *(_BYTE *)(v2 + 140) == 12 );
      result = *(_QWORD *)v2;
    }
    else
    {
      result = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        return result;
    }
    v5 = *(_QWORD *)(result + 96);
    if ( !a2 || (result = 1, (*(_BYTE *)(v5 + 176) & 1) == 0) && (*(_QWORD *)(v5 + 16) || !*(_QWORD *)(v5 + 8)) )
    {
      if ( !*(_QWORD *)(v5 + 24) )
        return (*(_BYTE *)(v5 + 179) & 0x40) != 0;
      result = 1;
      if ( (*(_BYTE *)(v5 + 177) & 2) != 0 )
        return (*(_BYTE *)(v5 + 179) & 0x40) != 0;
    }
  }
  return result;
}

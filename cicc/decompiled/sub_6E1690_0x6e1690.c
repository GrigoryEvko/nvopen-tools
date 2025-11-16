// Function: sub_6E1690
// Address: 0x6e1690
//
__int64 __fastcall sub_6E1690(__int64 a1)
{
  __int64 i; // rbx
  __int64 result; // rax

  for ( i = *(_QWORD *)(a1 + 88); i; i = *(_QWORD *)(i + 48) )
  {
    while ( *(_BYTE *)(i + 8) )
    {
      i = *(_QWORD *)(i + 48);
      if ( !i )
        return result;
    }
    result = sub_875E10(*(_QWORD *)i, *(_QWORD *)(i + 16), i + 32, 1, *(_QWORD *)(i + 24));
    *(_BYTE *)(i + 8) = 1;
  }
  return result;
}

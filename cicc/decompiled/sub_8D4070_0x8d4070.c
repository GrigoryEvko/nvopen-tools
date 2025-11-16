// Function: sub_8D4070
// Address: 0x8d4070
//
__int64 __fastcall sub_8D4070(__int64 a1)
{
  __int64 v1; // r12
  __int64 i; // rax
  __int64 v3; // rax

  v1 = a1;
  if ( !sub_8D3410(a1) )
    return 0;
  while ( 1 )
  {
    for ( i = v1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( (*(_BYTE *)(i + 169) & 2) != 0 )
      break;
    v3 = sub_8D4050(v1);
    v1 = v3;
    if ( !v3 || !sub_8D3410(v3) )
      return 0;
  }
  return 1;
}

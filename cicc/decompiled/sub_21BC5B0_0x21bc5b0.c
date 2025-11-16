// Function: sub_21BC5B0
// Address: 0x21bc5b0
//
__int64 __fastcall sub_21BC5B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r15
  unsigned __int64 v4; // rax
  __int64 v5; // r13
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 i; // r14
  __int64 v9; // rdi

  v2 = *(_QWORD *)(a2 + 80);
  v3 = *(_QWORD *)(v2 + 8);
  v4 = sub_157EBA0(v2 - 24);
  if ( v3 == a2 + 72 )
    return 0;
  v5 = v4;
  result = 0;
  do
  {
    if ( !v3 )
      BUG();
    v7 = *(_QWORD *)(v3 + 24);
    for ( i = v3 + 16; i != v7; result = 1 )
    {
      while ( 1 )
      {
        v9 = v7;
        v7 = *(_QWORD *)(v7 + 8);
        if ( *(_BYTE *)(v9 - 8) == 53 && *(_BYTE *)(*(_QWORD *)(v9 - 48) + 16LL) == 13 )
          break;
        if ( i == v7 )
          goto LABEL_10;
      }
      sub_15F22F0((_QWORD *)(v9 - 24), v5);
    }
LABEL_10:
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( a2 + 72 != v3 );
  return result;
}

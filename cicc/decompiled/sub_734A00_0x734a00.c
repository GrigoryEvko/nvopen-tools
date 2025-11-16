// Function: sub_734A00
// Address: 0x734a00
//
__int64 __fastcall sub_734A00(__int64 a1)
{
  _QWORD *i; // rbx
  __int64 v3; // rax

  if ( *(_BYTE *)(a1 + 140) != 7 )
    return 0;
  for ( i = **(_QWORD ***)(a1 + 168); i; i = (_QWORD *)*i )
  {
    while ( 1 )
    {
      v3 = i[5];
      if ( v3 )
      {
        if ( *(_BYTE *)(v3 + 24) == 10 )
          break;
      }
      i = (_QWORD *)*i;
      if ( !i )
        return 0;
    }
    sub_7347F0(*(_QWORD **)(v3 + 64));
    i[5] = 0;
  }
  return 0;
}

// Function: sub_14C4490
// Address: 0x14c4490
//
__int64 __fastcall sub_14C4490(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v5; // r13
  __int64 v6; // rax

  v3 = *(_QWORD *)(a1 + 8);
  if ( !v3 )
    return 0;
  v5 = 0;
  do
  {
    while ( 1 )
    {
      v6 = sub_1648700(v3);
      if ( (unsigned __int8)(*(_BYTE *)(v6 + 16) - 60) <= 0xCu && a3 == *(_QWORD *)v6 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        return v5;
    }
    if ( v5 )
      return 0;
    v3 = *(_QWORD *)(v3 + 8);
    v5 = v6;
  }
  while ( v3 );
  return v5;
}

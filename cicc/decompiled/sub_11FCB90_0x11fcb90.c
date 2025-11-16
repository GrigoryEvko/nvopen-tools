// Function: sub_11FCB90
// Address: 0x11fcb90
//
__int64 __fastcall sub_11FCB90(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v4; // r12
  _BYTE *v5; // rdi
  __int64 v6; // rax

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 )
    return 0;
  v4 = 0;
  do
  {
    while ( 1 )
    {
      v5 = *(_BYTE **)(v2 + 24);
      if ( *v5 == 85 )
        break;
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        return v4;
    }
    v6 = sub_B491C0((__int64)v5);
    v2 = *(_QWORD *)(v2 + 8);
    v4 += a1 == v6;
  }
  while ( v2 );
  return v4;
}

// Function: sub_B2BA20
// Address: 0xb2ba20
//
__int64 __fastcall sub_B2BA20(__int64 a1, unsigned __int8 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v5; // rdi

  v3 = a1 + 72;
  v4 = *(_QWORD *)(a1 + 80);
  if ( v4 != a1 + 72 )
  {
    do
    {
      v5 = v4 - 24;
      if ( !v4 )
        v5 = 0;
      sub_AA4880(v5, a2);
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( v3 != v4 );
  }
  *(_BYTE *)(a1 + 128) = a2;
  return a2;
}

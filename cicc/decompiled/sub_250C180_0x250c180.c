// Function: sub_250C180
// Address: 0x250c180
//
__int64 __fastcall sub_250C180(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  unsigned int v3; // r8d

  v2 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 <= 0x15u )
    return 1;
  if ( v2 > 0x1Cu )
  {
    LOBYTE(v3) = a2 == sub_B43CB0(a1);
    return v3;
  }
  v3 = 0;
  if ( v2 != 22 )
    return v3;
  LOBYTE(v3) = *(_QWORD *)(a1 + 24) == a2;
  return v3;
}

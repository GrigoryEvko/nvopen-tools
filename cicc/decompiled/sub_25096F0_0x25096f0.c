// Function: sub_25096F0
// Address: 0x25096f0
//
__int64 __fastcall sub_25096F0(_QWORD *a1)
{
  unsigned __int64 v1; // r8
  unsigned __int8 v2; // al

  v1 = *a1 & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*a1 & 3LL) == 3 )
    v1 = *(_QWORD *)(v1 + 24);
  v2 = *(_BYTE *)v1;
  if ( !*(_BYTE *)v1 )
    return v1;
  if ( v2 == 22 )
    return *(_QWORD *)(v1 + 24);
  if ( v2 <= 0x1Cu )
    return 0;
  else
    return sub_B43CB0(v1);
}

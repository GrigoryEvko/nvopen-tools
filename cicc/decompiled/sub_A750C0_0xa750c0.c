// Function: sub_A750C0
// Address: 0xa750c0
//
__int64 __fastcall sub_A750C0(__int64 a1)
{
  int v1; // edx
  unsigned int v2; // ecx
  unsigned __int8 v3; // al
  __int64 v5; // rax
  int v6; // edx

  v1 = *(unsigned __int8 *)(a1 + 8);
  v2 = v1 - 17;
  v3 = *(_BYTE *)(a1 + 8);
  if ( (unsigned int)(v1 - 17) <= 1 )
    v3 = *(_BYTE *)(**(_QWORD **)(a1 + 16) + 8LL);
  if ( v3 <= 3u || v3 == 5 || (v3 & 0xFD) == 4 )
    return 1;
  if ( (_BYTE)v1 != 15 )
  {
    if ( (_BYTE)v1 == 16 )
    {
      do
      {
        a1 = *(_QWORD *)(a1 + 24);
        LOBYTE(v1) = *(_BYTE *)(a1 + 8);
      }
      while ( (_BYTE)v1 == 16 );
      v2 = (unsigned __int8)v1 - 17;
    }
    if ( v2 <= 1 )
      LOBYTE(v1) = *(_BYTE *)(**(_QWORD **)(a1 + 16) + 8LL);
    return (unsigned __int8)v1 <= 3u || (_BYTE)v1 == 5 || (v1 & 0xFD) == 4;
  }
  if ( (*(_BYTE *)(a1 + 9) & 4) == 0 )
    return 0;
  if ( !(unsigned __int8)sub_BCB420(a1) )
    return 0;
  v5 = *(_QWORD *)(a1 + 16);
  v6 = *(unsigned __int8 *)(*(_QWORD *)v5 + 8LL);
  if ( (unsigned int)(v6 - 17) <= 1 )
    LOBYTE(v6) = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v5 + 16LL) + 8LL);
  return (unsigned __int8)v6 <= 3u || (_BYTE)v6 == 5 || (v6 & 0xFD) == 4;
}

// Function: sub_BDB700
// Address: 0xbdb700
//
__int64 __fastcall sub_BDB700(__int64 a1)
{
  int v1; // eax
  unsigned int v2; // r8d

  v1 = *(unsigned __int8 *)(a1 + 8);
  if ( (unsigned int)(v1 - 17) <= 1 )
    LOBYTE(v1) = *(_BYTE *)(**(_QWORD **)(a1 + 16) + 8LL);
  v2 = 1;
  if ( (unsigned __int8)v1 > 3u && (_BYTE)v1 != 5 )
    LOBYTE(v2) = (v1 & 0xFD) == 4;
  return v2;
}

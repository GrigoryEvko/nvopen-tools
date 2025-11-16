// Function: sub_9C3C60
// Address: 0x9c3c60
//
__int64 __fastcall sub_9C3C60(int a1, __int64 a2)
{
  int v2; // eax

  v2 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned int)(v2 - 17) <= 1 )
    LOBYTE(v2) = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
  if ( (unsigned __int8)v2 <= 3u || (_BYTE)v2 == 5 || (v2 & 0xFD) == 4 )
    return a1 == 0 ? 12 : -1;
  else
    return 0xFFFFFFFFLL;
}

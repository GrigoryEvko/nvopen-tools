// Function: sub_3007030
// Address: 0x3007030
//
__int64 __fastcall sub_3007030(__int64 a1)
{
  __int64 v1; // rcx
  int v2; // eax
  unsigned int v3; // r8d

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(unsigned __int8 *)(v1 + 8);
  if ( (unsigned int)(v2 - 17) <= 1 )
    LOBYTE(v2) = *(_BYTE *)(**(_QWORD **)(v1 + 16) + 8LL);
  v3 = 1;
  if ( (unsigned __int8)v2 > 3u && (_BYTE)v2 != 5 )
    LOBYTE(v3) = (v2 & 0xFD) == 4;
  return v3;
}

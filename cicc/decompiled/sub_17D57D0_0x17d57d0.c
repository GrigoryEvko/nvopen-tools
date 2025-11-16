// Function: sub_17D57D0
// Address: 0x17d57d0
//
__int64 *__fastcall sub_17D57D0(__int128 a1, int a2)
{
  if ( (*(_BYTE *)(*((_QWORD *)&a1 + 1) + 23LL) & 0x40) != 0 )
    *((_QWORD *)&a1 + 1) = *(_QWORD *)(*(_QWORD *)(*((_QWORD *)&a1 + 1) - 8LL) + 24LL * a2);
  else
    *((_QWORD *)&a1 + 1) = *(_QWORD *)(*((_QWORD *)&a1 + 1)
                                     - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF)
                                     + 24LL * a2);
  return sub_17D4DA0(a1);
}

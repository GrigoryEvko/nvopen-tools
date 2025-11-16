// Function: sub_884800
// Address: 0x884800
//
_QWORD *__fastcall sub_884800(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 *v7; // r9

  result = (_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40);
  *((_BYTE *)result + 7) &= ~8u;
  if ( result[57] )
  {
    if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
      sub_8646E0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), 0);
    sub_8600D0(0xEu, -1, 0, a1);
    v3 = dword_4F04C40;
    sub_8845B0(dword_4F04C40);
    result = sub_863FC0(v3, 0xFFFFFFFFLL, v4, v5, v6, v7);
    if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
      return sub_866010();
  }
  return result;
}

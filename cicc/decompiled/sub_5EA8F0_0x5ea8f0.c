// Function: sub_5EA8F0
// Address: 0x5ea8f0
//
__int64 __fastcall sub_5EA8F0(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 *v6; // rbx
  __int64 v7; // rdi

  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  v5 = v4;
  while ( 1 )
  {
    v6 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)v5 + 96LL) + 56LL);
    if ( v6 )
      break;
LABEL_11:
    if ( (*(_BYTE *)(v5 + 89) & 4) != 0 )
    {
      v5 = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 32LL);
      if ( v5 )
        continue;
    }
    sub_8646E0(v4, 0);
    sub_8600D0(1, 0xFFFFFFFFLL, *(_QWORD *)(a1 + 152), 0);
    *(_BYTE *)(unk_4F04C68 + 776LL * dword_4F04C64 + 11) |= 0x40u;
    BUG();
  }
  while ( a1 != *(_QWORD *)(v6[2] + 88) )
  {
    v6 = (__int64 *)*v6;
    if ( !v6 )
      goto LABEL_11;
  }
  sub_8646E0(v4, 0);
  sub_8600D0(1, 0xFFFFFFFFLL, *(_QWORD *)(a1 + 152), 0);
  *(_BYTE *)(unk_4F04C68 + 776LL * dword_4F04C64 + 11) |= 0x40u;
  v7 = v6[3];
  if ( v7 )
    sub_886000(v7);
  *(_BYTE *)a2 &= ~0x20u;
  *(_QWORD *)(a2 + 8) = 0;
  *((_BYTE *)v6 + 184) &= ~8u;
  if ( v3 )
  {
    sub_625150(a1, v3, 0);
    sub_7AEB40(v3);
  }
  sub_863FC0();
  return sub_866010();
}

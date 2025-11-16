// Function: sub_7D2400
// Address: 0x7d2400
//
_QWORD *__fastcall sub_7D2400(_QWORD *a1)
{
  __int64 v1; // r12
  __int64 v3; // rax
  char v4; // dl
  __int64 v5; // [rsp+8h] [rbp-68h] BYREF
  _BYTE v6[16]; // [rsp+10h] [rbp-60h] BYREF
  char v7; // [rsp+20h] [rbp-50h]
  char v8; // [rsp+21h] [rbp-4Fh]
  __int64 v9; // [rsp+28h] [rbp-48h]

  v1 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 208);
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v1 + 96LL) + 180LL) & 2) == 0 )
    return a1;
  sub_878710(*a1, v6);
  if ( (v8 & 0x40) == 0 )
  {
    v7 &= ~0x80u;
    v9 = 0;
  }
  if ( !(unsigned int)sub_886B00(v1, (unsigned int)v6, 2048, 1, 0, 0, 0, 0, 0, 0, (__int64)&v5, 1) )
    return a1;
  v3 = v5;
  v4 = *(_BYTE *)(v5 + 80);
  if ( v4 == 16 )
  {
    v3 = **(_QWORD **)(v5 + 88);
    v4 = *(_BYTE *)(v3 + 80);
  }
  if ( v4 == 24 )
    v3 = *(_QWORD *)(v3 + 88);
  return *(_QWORD **)(v3 + 88);
}

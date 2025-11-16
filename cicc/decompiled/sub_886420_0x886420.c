// Function: sub_886420
// Address: 0x886420
//
_QWORD *__fastcall sub_886420(__int64 a1, __int64 a2, int a3, int a4)
{
  _QWORD *v5; // r12
  char v7; // dl
  __int64 v8; // rax
  int v9[9]; // [rsp+Ch] [rbp-24h] BYREF

  v9[0] = a4;
  if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * a3 + 4) == 1 && (*(_BYTE *)(a2 + 17) & 0x20) == 0 )
  {
    v7 = *(_BYTE *)(a1 + 140);
    if ( v7 == 12 )
    {
      v8 = a1;
      do
      {
        v8 = *(_QWORD *)(v8 + 160);
        v7 = *(_BYTE *)(v8 + 140);
      }
      while ( v7 == 12 );
    }
    if ( v7 )
      sub_684B30(0xE7u, (_DWORD *)(a2 + 8));
  }
  v5 = sub_87EBB0(3u, *(_QWORD *)a2, (_QWORD *)(a2 + 8));
  *((_BYTE *)v5 + 81) = *(_BYTE *)(a2 + 17) & 0x20 | *((_BYTE *)v5 + 81) & 0xDF;
  *(_BYTE *)(a2 + 16) &= ~1u;
  *(_QWORD *)(a2 + 24) = v5;
  v5[11] = a1;
  sub_885620((__int64)v5, a3, v9);
  sub_881ED0((__int64)v5, a3, v9[0]);
  return v5;
}

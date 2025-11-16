// Function: sub_921CE0
// Address: 0x921ce0
//
__int64 __fastcall sub_921CE0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rax
  unsigned __int16 v6; // bx
  __int64 v7; // rax

  if ( *(char *)(a2 + 142) >= 0 && *(_BYTE *)(a2 + 140) == 12 )
    v5 = (unsigned int)sub_8D4AB0(a2);
  else
    v5 = *(unsigned int *)(a2 + 136);
  v6 = 0;
  if ( v5 )
  {
    _BitScanReverse64(&v5, v5);
    LOBYTE(v6) = 63 - (v5 ^ 0x3F);
    HIBYTE(v6) = 1;
  }
  v7 = sub_91A390(*(_QWORD *)(a1 + 32) + 8LL, a2, 0, a4);
  return sub_921B80(a1, v7, a3, v6, 0);
}

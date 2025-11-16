// Function: sub_921D70
// Address: 0x921d70
//
__int64 __fastcall sub_921D70(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  unsigned __int64 v5; // rax
  unsigned __int16 v6; // bx
  int v7; // eax
  __int64 v8; // rax

  v4 = a3;
  if ( *(char *)(a2 + 142) >= 0 && *(_BYTE *)(a2 + 140) == 12 )
    v5 = (unsigned int)sub_8D4AB0(a2);
  else
    v5 = *(unsigned int *)(a2 + 136);
  v6 = 0;
  if ( v5 )
  {
    _BitScanReverse64(&v5, v5);
    v7 = v5 ^ 0x3F;
    a3 = (unsigned int)(63 - v7);
    LOBYTE(v6) = 63 - v7;
    HIBYTE(v6) = 1;
  }
  v8 = sub_91A3A0(*(_QWORD *)(a1 + 32) + 8LL, a2, a3, a4);
  return sub_921B80(a1, v8, v4, v6, 0);
}

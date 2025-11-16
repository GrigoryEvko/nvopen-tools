// Function: sub_1309680
// Address: 0x1309680
//
__int64 __fastcall sub_1309680(__int64 a1, _QWORD *a2, unsigned __int64 a3, unsigned __int8 a4)
{
  __int64 v5; // r15
  __int64 v6; // r14
  unsigned __int64 v7; // r10
  int v8; // r8d
  int v9; // r9d
  unsigned int v10; // r8d
  char v12; // cl
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // [rsp-8h] [rbp-58h]
  unsigned __int8 v16; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v17; // [rsp+Fh] [rbp-41h]
  _BYTE v18[49]; // [rsp+1Fh] [rbp-31h] BYREF

  v5 = qword_50579C0[*a2 & 0xFFFLL];
  v6 = qword_505FA40[(unsigned __int8)(*a2 >> 20)];
  v7 = a2[2] & 0xFFFFFFFFFFFFF000LL;
  v8 = a3 + unk_50607C0;
  if ( a3 > 0x1000 )
  {
    if ( a3 > 0x7000000000000000LL )
    {
      v9 = 232;
    }
    else
    {
      v12 = 7;
      _BitScanReverse64((unsigned __int64 *)&v13, 2 * a3 - 1);
      if ( (unsigned int)v13 >= 7 )
        v12 = v13;
      v14 = (((-1LL << (v12 - 3)) & (a3 - 1)) >> (v12 - 3)) & 3;
      if ( (unsigned int)v13 < 6 )
        LODWORD(v13) = 6;
      v9 = v14 + 4 * v13 - 23;
    }
  }
  else
  {
    v9 = byte_5060800[(a3 + 7) >> 3];
  }
  v18[0] = 0;
  v10 = sub_130B660(a1, (int)v5 + 10648, (_DWORD)a2, v7, v8, v9, a4, (__int64)v18);
  if ( v18[0] )
  {
    v17 = v10;
    sub_1314D40(a1, v5, v15);
    v10 = v17;
  }
  if ( !(_BYTE)v10 )
  {
    if ( a4 && unk_4C6F0DC )
    {
      memset((void *)(v6 + a2[1]), 0, ((v6 + a2[1] + 4096) & 0xFFFFFFFFFFFFF000LL) - (v6 + a2[1]));
      LOBYTE(v10) = 0;
    }
    v16 = v10;
    sub_1314FE0(a1, v5, a2, v6);
    return v16;
  }
  return v10;
}

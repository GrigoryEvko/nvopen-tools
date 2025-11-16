// Function: sub_2C13BB0
// Address: 0x2c13bb0
//
void __fastcall sub_2C13BB0(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  unsigned __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r10
  __int64 v9; // rax
  _BYTE *v10; // rsi
  unsigned __int16 v11; // cx
  __int64 v12; // [rsp+8h] [rbp-78h]
  __int64 v13; // [rsp+8h] [rbp-78h]
  char v14; // [rsp+10h] [rbp-70h]
  char v15; // [rsp+17h] [rbp-69h]
  unsigned int **v16; // [rsp+18h] [rbp-68h]
  __int64 v17[4]; // [rsp+20h] [rbp-60h] BYREF
  char v18; // [rsp+40h] [rbp-40h]
  char v19; // [rsp+41h] [rbp-3Fh]

  v3 = *(_QWORD *)(a1 + 96);
  v12 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL);
  v15 = *(_BYTE *)(a1 + 104);
  _BitScanReverse64(&v4, 1LL << (*(_WORD *)(v3 + 2) >> 1));
  v14 = 63 - (v4 ^ 0x3F);
  v16 = *(unsigned int ***)(a2 + 904);
  v17[0] = *(_QWORD *)(a1 + 88);
  if ( v17[0] )
    sub_2AAAFA0(v17);
  v5 = 0;
  sub_2BF1A90(a2, (__int64)v17);
  sub_9C6650(v17);
  if ( *(_BYTE *)(a1 + 106) )
  {
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 56) - 1));
    if ( v5 )
    {
      v6 = sub_2BFB640(a2, v5, 0);
      v5 = v6;
      if ( *(_BYTE *)(a1 + 105) )
      {
        v19 = 1;
        v18 = 3;
        v17[0] = (__int64)"reverse";
        v5 = sub_B37000(v16, v6, (__int64)v17);
      }
    }
  }
  v7 = sub_2BFB640(a2, v12, 0);
  v8 = v7;
  if ( *(_BYTE *)(a1 + 105) )
  {
    v19 = 1;
    v18 = 3;
    v17[0] = (__int64)"reverse";
    v8 = sub_B37000(v16, v7, (__int64)v17);
  }
  v13 = v8;
  v9 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), v15);
  if ( v15 )
  {
    if ( v5 )
    {
      v10 = (_BYTE *)sub_B34CE0((__int64)v16, v13, v9, v14, v5);
    }
    else
    {
      LOBYTE(v11) = v14;
      HIBYTE(v11) = 1;
      v10 = (_BYTE *)sub_2463EC0((__int64 *)v16, v13, v9, v11, 0);
    }
  }
  else
  {
    v10 = (_BYTE *)sub_B34E90((__int64)v16, v13, v9, v14, v5);
  }
  sub_2BF08A0(a2, v10, (_BYTE *)v3);
}

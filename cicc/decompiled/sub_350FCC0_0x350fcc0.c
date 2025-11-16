// Function: sub_350FCC0
// Address: 0x350fcc0
//
__int64 __fastcall sub_350FCC0(unsigned __int16 a1)
{
  __int64 v1; // rax
  char v2; // cl
  __int64 v3; // rax
  __int64 v5; // r12
  int v6; // eax
  __int64 v7; // rax
  char v8; // cl
  __int64 v9; // rax
  __int64 v10; // rcx
  bool v11; // dl
  unsigned __int64 v12; // rsi
  __int64 v13; // rax
  bool v14; // si
  __int64 v15; // [rsp+8h] [rbp-28h]
  __int64 v16; // [rsp+10h] [rbp-20h] BYREF
  char v17; // [rsp+18h] [rbp-18h]

  if ( (unsigned __int16)(a1 - 17) > 0xD3u )
  {
    if ( a1 > 1u && (unsigned __int16)(a1 - 504) > 7u )
    {
      v1 = 16LL * (a1 - 1);
      v2 = byte_444C4A0[v1 + 8];
      v3 = *(_QWORD *)&byte_444C4A0[v1];
      v17 = v2;
      v16 = v3;
      return (sub_CA1930(&v16) << 32) | 1;
    }
LABEL_11:
    BUG();
  }
  v5 = a1 - 1;
  v6 = (unsigned __int16)word_4456580[v5];
  if ( (unsigned __int16)v6 <= 1u || (unsigned __int16)(word_4456580[v5] - 504) <= 7u )
    goto LABEL_11;
  v7 = 16LL * (v6 - 1);
  v8 = byte_444C4A0[v7 + 8];
  v9 = *(_QWORD *)&byte_444C4A0[v7];
  v17 = v8;
  v16 = v9;
  v10 = (sub_CA1930(&v16) << 29) & 0x1FFFFFFFE0000000LL;
  v11 = word_4456340[v5] == 1 && (unsigned __int16)(a1 - 176) > 0x34u;
  if ( v11 )
  {
    v14 = 0;
    LOBYTE(v13) = 0;
  }
  else
  {
    LODWORD(v15) = word_4456340[v5];
    BYTE4(v15) = (unsigned __int16)(a1 - 176) <= 0x34u;
    v12 = sub_350F8F0(v15, 8 * v10 + 1);
    v11 = v12 & 1;
    v13 = (v12 >> 1) & 1;
    v10 = v12 >> 3;
    v14 = (v12 & 4) != 0;
  }
  return (8 * v10) | (4LL * v14) | v11 | (2LL * (unsigned __int8)v13);
}

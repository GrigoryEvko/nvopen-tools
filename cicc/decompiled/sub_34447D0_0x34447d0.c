// Function: sub_34447D0
// Address: 0x34447d0
//
_BOOL8 __fastcall sub_34447D0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int16 v4; // bx
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rsi
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned __int16 v19; // [rsp+10h] [rbp-40h] BYREF
  __int64 v20; // [rsp+18h] [rbp-38h]
  __int64 v21; // [rsp+20h] [rbp-30h]
  __int64 v22; // [rsp+28h] [rbp-28h]

  v4 = a2;
  v17 = a2;
  v18 = a3;
  if ( (_WORD)a2 )
  {
    if ( (unsigned __int16)(a2 - 17) <= 0xD3u )
    {
      v20 = 0;
      v4 = word_4456580[(unsigned __int16)a2 - 1];
      v19 = v4;
      if ( !v4 )
        goto LABEL_5;
      goto LABEL_17;
    }
    goto LABEL_3;
  }
  if ( !sub_30070B0((__int64)&v17) )
  {
LABEL_3:
    v5 = v18;
    goto LABEL_4;
  }
  v4 = sub_3009970((__int64)&v17, a2, v14, v15, v16);
LABEL_4:
  v19 = v4;
  v20 = v5;
  if ( !v4 )
  {
LABEL_5:
    v6 = sub_3007260((__int64)&v19);
    v7 = v8;
    v21 = v6;
    LODWORD(v8) = v6;
    v22 = v7;
    goto LABEL_6;
  }
LABEL_17:
  if ( v4 == 1 || (unsigned __int16)(v4 - 504) <= 7u )
    BUG();
  v8 = *(_QWORD *)&byte_444C4A0[16 * v4 - 16];
LABEL_6:
  if ( (_WORD)v17 == 1 )
  {
    if ( (*(_BYTE *)(a1 + 6970) & 0xFB) == 0 && (*(_BYTE *)(a1 + 6971) & 0xFB) == 0 )
    {
      v10 = 1;
      goto LABEL_22;
    }
    return 0;
  }
  if ( !(_WORD)v17 )
    return 0;
  v10 = (unsigned __int16)v17;
  v11 = (unsigned __int16)v17 + 14LL;
  if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v17 + 112) )
    return 0;
  v12 = a1 + 500LL * (unsigned __int16)v17;
  if ( (*(_BYTE *)(v12 + 6470) & 0xFB) != 0
    || !*(_QWORD *)(a1 + 8 * v11)
    || (*(_BYTE *)(v12 + 6471) & 0xFB) != 0
    || !*(_QWORD *)(a1 + 8 * v11) )
  {
    return 0;
  }
LABEL_22:
  if ( (*(_BYTE *)(a1 + 500 * v10 + 6606) & 0xFB) != 0
    || (_DWORD)v8 != 8
    && ((_WORD)v17 != 1 && !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v17 + 112)
     || (*(_BYTE *)(a1 + 500 * v10 + 6472) & 0xFB) != 0) )
  {
    return 0;
  }
  if ( (_WORD)v17 != 1 && !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v17 + 112) )
    return 0;
  v13 = *(_BYTE *)(a1 + 500 * v10 + 6600);
  return v13 <= 1u || v13 == 4;
}

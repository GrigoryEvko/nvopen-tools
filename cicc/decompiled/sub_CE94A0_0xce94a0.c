// Function: sub_CE94A0
// Address: 0xce94a0
//
unsigned __int16 __fastcall sub_CE94A0(__int64 a1, unsigned int a2)
{
  unsigned __int16 result; // ax
  __int64 v3; // rax
  __int64 v4; // r9
  unsigned __int8 v5; // dl
  bool v6; // di
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  char v15; // cl
  unsigned __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-20h] BYREF
  __int64 v18[3]; // [rsp+8h] [rbp-18h] BYREF

  v17 = *(_QWORD *)(a1 + 72);
  v18[0] = sub_A74490(&v17, a2);
  result = sub_A73690(v18);
  if ( HIBYTE(result) )
    return result;
  if ( !*(_QWORD *)(a1 + 48) && (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return 0;
  v3 = sub_B91F50(a1, "callalign", 9u);
  v4 = v3;
  if ( !v3 )
    return 0;
  v5 = *(_BYTE *)(v3 - 16);
  v6 = (v5 & 2) != 0;
  v7 = (v5 & 2) != 0 ? *(unsigned int *)(v3 - 24) : (*(_WORD *)(v3 - 16) >> 6) & 0xFu;
  if ( (int)v7 <= 0 )
    return 0;
  v8 = 8 * v7;
  v9 = 0;
  v10 = v4 - 8LL * ((v5 >> 2) & 0xF) - 16;
  while ( 1 )
  {
    v12 = v10;
    if ( v6 )
      v12 = *(_QWORD *)(v4 - 32);
    v13 = *(_QWORD *)(v12 + v9);
    if ( *(_BYTE *)v13 == 1 )
    {
      v14 = *(_QWORD *)(v13 + 136);
      if ( *(_BYTE *)v14 == 17 )
        break;
    }
LABEL_11:
    v9 += 8;
    if ( v8 == v9 )
      return 0;
  }
  if ( *(_DWORD *)(v14 + 32) > 0x40u )
    v11 = **(_QWORD **)(v14 + 24);
  else
    v11 = *(_QWORD *)(v14 + 24);
  if ( WORD1(v11) != a2 )
  {
    if ( WORD1(v11) > a2 )
      return 0;
    goto LABEL_11;
  }
  v15 = -1;
  if ( (_WORD)v11 )
  {
    _BitScanReverse64(&v16, (unsigned __int16)v11);
    v15 = 63 - (v16 ^ 0x3F);
  }
  LOBYTE(result) = v15;
  HIBYTE(result) = 1;
  return result;
}

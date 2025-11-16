// Function: sub_12F9930
// Address: 0x12f9930
//
unsigned __int64 __fastcall sub_12F9930(unsigned __int64 *a1, char a2)
{
  __int64 v3; // rsi
  char v4; // r10
  unsigned __int64 v5; // r8
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // r9

  v3 = a1[1];
  v4 = a2;
  v5 = *a1;
  v6 = HIWORD(v3) & 0x7FFF;
  v7 = v3 & 0xFFFFFFFFFFFFLL;
  v8 = 16431 - v6;
  if ( 16431 - v6 >= 0 )
  {
    if ( v8 > 48 )
    {
      v10 = 0;
      if ( v4 && v6 | v5 | v7 )
        unk_4F968EA |= 1u;
      return v10;
    }
    v9 = v7 | 0x1000000000000LL;
    v10 = v9 >> v8;
    if ( !v4 || !v5 && v10 << v8 == v9 )
    {
LABEL_7:
      if ( v3 < 0 )
        return -(__int64)v10;
      return v10;
    }
LABEL_6:
    unk_4F968EA |= 1u;
    goto LABEL_7;
  }
  if ( v8 >= -14 )
  {
    v10 = ((v7 | 0x1000000000000LL) << (BYTE6(v3) - 47)) | (v5 >> (47 - BYTE6(v3)));
    if ( !v4 || !(v5 << (BYTE6(v3) - 47)) )
      goto LABEL_7;
    goto LABEL_6;
  }
  if ( v3 == 0xC03E000000000000LL && v5 <= 0x1FFFFFFFFFFFFLL )
  {
    if ( !v5 || !v4 )
      return 0x8000000000000000LL;
    v10 = 0x8000000000000000LL;
    unk_4F968EA |= 1u;
    return v10;
  }
  sub_12F9B70(16, v3, v6, 16431, v5);
  return 0x8000000000000000LL;
}

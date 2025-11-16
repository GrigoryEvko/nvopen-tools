// Function: sub_12F9910
// Address: 0x12f9910
//
__int64 __fastcall sub_12F9910(unsigned __int64 *a1, unsigned __int8 a2)
{
  unsigned int v2; // edx
  __int64 v3; // rsi
  unsigned __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // rdx
  unsigned __int64 v9; // rdi
  __int64 result; // rax

  v2 = a2;
  v3 = a1[1];
  v4 = *a1;
  v5 = v2;
  v6 = HIWORD(v3) & 0x7FFF;
  v7 = v3 & 0xFFFFFFFFFFFFLL;
  v8 = 16431 - v6;
  if ( 16431 - v6 < 0 )
  {
    if ( v8 >= -15 && v3 >= 0 )
    {
      result = ((v7 | 0x1000000000000LL) << (BYTE6(v3) - 47)) | (v4 >> (47 - BYTE6(v3)));
      if ( (_BYTE)v5 && v4 << (BYTE6(v3) - 47) )
        goto LABEL_7;
      return result;
    }
    return sub_12FA8C0(v7, v3, v8, v6, v4, v5);
  }
  if ( v8 > 48 )
  {
    result = 0;
    if ( (_BYTE)v5 )
    {
      result = v6 | v4 | v7;
      if ( result )
      {
        unk_4F968EA |= 1u;
        return 0;
      }
    }
    return result;
  }
  if ( v3 < 0 )
    return sub_12FA8C0(v7, v3, v8, v6, v4, v5);
  v9 = v7 | 0x1000000000000LL;
  result = v9 >> v8;
  if ( (_BYTE)v5 && (v4 || result << v8 != v9) )
LABEL_7:
    unk_4F968EA |= 1u;
  return result;
}

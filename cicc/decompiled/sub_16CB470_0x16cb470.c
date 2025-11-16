// Function: sub_16CB470
// Address: 0x16cb470
//
unsigned __int64 __fastcall sub_16CB470(unsigned __int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rcx
  unsigned __int64 v3; // r8
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  bool v6; // cf
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rsi
  unsigned __int64 result; // rax
  unsigned __int64 v10; // rcx
  int v11; // ecx

  v2 = (unsigned int)a2 * HIDWORD(a1);
  v3 = HIDWORD(a2) * HIDWORD(a1);
  v4 = (unsigned int)a1 * HIDWORD(a2);
  v5 = (unsigned int)a2 * (unsigned __int64)(unsigned int)a1;
  v6 = __CFADD__(v2 << 32, v5);
  v7 = (v2 << 32) + v5;
  v8 = v7 + (v4 << 32);
  result = __CFADD__(v7, v4 << 32) + v3 + HIDWORD(v2) + HIDWORD(v4) + v6;
  if ( !result )
    return v8;
  _BitScanReverse64(&v10, result);
  v11 = v10 ^ 0x3F;
  if ( v11 )
    result = (v8 >> (64 - (unsigned __int8)v11)) | (result << v11);
  if ( _bittest64((const __int64 *)&v8, (unsigned int)(63 - v11)) )
  {
    if ( !++result )
      return 0x8000000000000000LL;
  }
  return result;
}

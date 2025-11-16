// Function: sub_F02E20
// Address: 0xf02e20
//
__int64 __fastcall sub_F02E20(unsigned int *a1, unsigned __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rsi
  __int64 v5; // rdx
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rax
  bool v8; // cf
  __int64 result; // rax

  v2 = *a1;
  if ( !a2 || (_DWORD)v2 == 0x80000000 )
    return a2;
  v4 = v2 * (unsigned int)a2;
  v5 = HIDWORD(v4) + v2 * HIDWORD(a2);
  if ( v5 < 0 )
    return -1;
  v6 = (unsigned __int64)v5 >> 31 << 32;
  v7 = ((v5 << 32) & 0x7FFFFFFF00000000LL | (unsigned __int64)(unsigned int)v4) >> 31;
  v8 = __CFADD__(v6, v7);
  result = v6 + v7;
  if ( v8 )
    return -1;
  return result;
}

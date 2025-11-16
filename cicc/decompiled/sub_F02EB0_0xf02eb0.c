// Function: sub_F02EB0
// Address: 0xf02eb0
//
__int64 __fastcall sub_F02EB0(unsigned int *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r8
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rcx
  __int64 result; // rax
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax
  bool v9; // cf

  v2 = *a1;
  if ( !a2 || (_DWORD)v2 == 0x80000000 )
    return a2;
  v3 = ((unsigned int)a2 >> 1) + (HIDWORD(a2) << 31);
  v4 = v3 % v2;
  v5 = v3 / v2;
  result = -1;
  if ( v5 <= 0xFFFFFFFF )
  {
    v7 = v5 << 32;
    v8 = ((unsigned int)((_DWORD)a2 << 31) | (v4 << 32)) / v2;
    v9 = __CFADD__(v7, v8);
    result = v7 + v8;
    if ( v9 )
      return -1;
  }
  return result;
}

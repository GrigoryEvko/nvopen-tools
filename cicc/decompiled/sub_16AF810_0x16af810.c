// Function: sub_16AF810
// Address: 0x16af810
//
__int64 __fastcall sub_16AF810(unsigned int *a1, unsigned __int64 a2)
{
  unsigned int v2; // ecx
  __int64 v3; // r8
  unsigned __int64 v4; // kr00_8
  unsigned __int64 v5; // r9
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rax
  bool v10; // cf
  unsigned __int64 v11; // rax

  v2 = *a1;
  if ( !a2 || v2 == 0x80000000 )
    return a2;
  v3 = -1;
  v4 = ((unsigned int)a2 >> 1) + (HIDWORD(a2) << 31);
  if ( v2 > HIDWORD(v4) )
  {
    v5 = v2;
    v6 = ((unsigned int)v4 | ((unsigned __int64)HIDWORD(v4) << 32)) % v2;
    v7 = ((unsigned int)v4 | ((unsigned __int64)HIDWORD(v4) << 32)) / v2;
    if ( v7 <= 0xFFFFFFFF )
    {
      v8 = v7 << 32;
      v9 = ((unsigned int)((_DWORD)a2 << 31) | (v6 << 32)) / v5;
      v10 = __CFADD__(v8, v9);
      v11 = v8 + v9;
      if ( !v10 )
        return v11;
    }
  }
  return v3;
}

// Function: sub_2EF2C60
// Address: 0x2ef2c60
//
unsigned __int64 __fastcall sub_2EF2C60(__int64 a1)
{
  unsigned __int64 v1; // rdx
  unsigned __int64 result; // rax
  char v3; // al
  unsigned __int64 v4; // rcx
  char v5; // di
  int v6; // esi
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax

  v1 = *(_QWORD *)(a1 + 24);
  result = -1;
  if ( (v1 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
  {
    v4 = v1 >> 3;
    v3 = *(_BYTE *)(a1 + 24);
    v5 = v3 & 2;
    if ( (v3 & 6) == 2 || (v3 & 1) != 0 )
    {
      v10 = HIWORD(v1);
      if ( !v5 )
        v10 = HIDWORD(v1);
      return (v10 + 7) >> 3;
    }
    else
    {
      v6 = (unsigned __int16)((unsigned int)v1 >> 8);
      v7 = v1;
      v8 = HIDWORD(v1);
      v9 = HIWORD(v7);
      if ( !v5 )
        LODWORD(v9) = v8;
      result = ((unsigned __int64)(unsigned int)(v6 * v9) + 7) >> 3;
      if ( (v4 & 1) != 0 )
        result |= 0x4000000000000000uLL;
    }
  }
  return result;
}

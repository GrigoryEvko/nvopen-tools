// Function: sub_1346A60
// Address: 0x1346a60
//
unsigned __int64 __fastcall sub_1346A60(unsigned __int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  int v3; // ecx
  unsigned int v4; // r8d
  int v5; // r12d
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r14

  v2 = a1;
  v3 = 0;
  a1 = (unsigned __int16)a1;
  v4 = HIWORD(v2);
  v5 = 0;
  do
  {
    v7 = a1;
    v8 = 10 * a1;
    a1 *= 10LL;
    if ( v7 <= 0xFFFF && v8 >= 0x10000 )
      v5 = v3;
    ++v3;
  }
  while ( v3 != 14 );
  v9 = v8 >> 16;
  v10 = v9;
  if ( v9 )
  {
    v11 = v9 % 0xA;
    if ( v9 == 10 * (v9 / 0xA) )
    {
      while ( 1 )
      {
        v12 = v10;
        v10 /= 0xAu;
        if ( v12 <= 9 )
          break;
        if ( __ROR8__(0xCCCCCCCCCCCCCCCDLL * v10, 1) > 0x1999999999999999uLL )
          goto LABEL_14;
      }
    }
    else
    {
LABEL_14:
      v11 = v10;
    }
  }
  else
  {
    v11 = 0;
  }
  v13 = sub_40E1DF(a2, 0x15u, "%u.", v4);
  v14 = v13;
  if ( v5 )
  {
    v14 = v5 + v13;
    memset((void *)(a2 + v13), 48, v5);
  }
  return sub_40E1DF(a2 + v14, 21 - v14, (char *)"%lu", v11);
}

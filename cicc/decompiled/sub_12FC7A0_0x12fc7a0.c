// Function: sub_12FC7A0
// Address: 0x12fc7a0
//
unsigned __int64 __fastcall sub_12FC7A0(char a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 v4; // r9
  __int64 v5; // r10
  unsigned __int64 v6; // rax
  unsigned __int64 v8; // r8
  char v9; // r8
  char v10; // al
  __int64 v11; // rsi

  v4 = a4;
  if ( !a3 )
  {
    v5 = a2 - 113;
    v6 = 0;
    if ( !a4 )
    {
LABEL_3:
      a3 = v4;
      v4 = v6;
      goto LABEL_4;
    }
    a2 -= 64;
    a3 = a4;
    v4 = 0;
  }
  _BitScanReverse64(&v8, a3);
  v9 = v8 ^ 0x3F;
  v10 = v9 - 15;
  v11 = a2 - (char)(v9 - 15);
  v5 = v11;
  if ( (char)(v9 - 15) < 0 )
    return sub_12FC4A0(a1, v11, a3 >> (15 - v9), (v4 >> (15 - v9)) | (a3 << (v10 & 0x3F)), v4 << (v10 & 0x3F));
  if ( v9 != 15 )
  {
    v6 = v4 << v10;
    v4 = (a3 << (v9 - 15)) | (v4 >> (15 - v9));
    goto LABEL_3;
  }
LABEL_4:
  if ( (unsigned int)v5 <= 0x7FFC )
    return v4;
  else
    return sub_12FC4A0(a1, v5, a3, v4, 0);
}

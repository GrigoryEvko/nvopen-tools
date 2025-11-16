// Function: sub_12F9D30
// Address: 0x12f9d30
//
__int64 __fastcall sub_12F9D30(int a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  char v4; // r15
  int v5; // r14d
  unsigned __int64 v6; // rbx
  unsigned __int64 v8; // rbx
  int v9; // ecx
  bool v10; // al
  unsigned int v11; // r14d

  v3 = 64;
  v4 = unk_4F968EB;
  if ( (unk_4F968EB & 0xFB) != 0 )
  {
    v3 = 127;
    if ( unk_4F968EB != ((_BYTE)a1 == 0) + 2 )
      v3 = 0;
  }
  LOBYTE(v5) = a3 & 0x7F;
  if ( (unsigned int)a2 <= 0xFC )
  {
    v6 = a3 + (unsigned __int8)v3;
    goto LABEL_10;
  }
  if ( a2 >= 0 )
  {
    if ( a2 != 253 || (v6 = a3 + (unsigned __int8)v3, v6 > 0x7FFFFFFF) )
    {
      sub_12F9B70(5);
      return (a1 << 31) - ((unsigned int)((_BYTE)v3 == 0) - 2139095040);
    }
    goto LABEL_10;
  }
  v10 = unk_4C6F00D == 0 || a2 != -1;
  if ( v10 )
  {
    v11 = a3 != 0;
    if ( a2 < -30 )
      goto LABEL_16;
  }
  else
  {
    v10 = (unsigned __int64)(a3 + v3) <= 0x7FFFFFFF;
  }
  v11 = ((unsigned int)a3 >> -(char)a2) | ((_DWORD)a3 << a2 != 0);
LABEL_16:
  v6 = v3 + v11;
  v5 = v11 & 0x7F;
  if ( v5 && v10 )
  {
    v8 = v6 >> 7;
    sub_12F9B70(2);
    LODWORD(a2) = 0;
    goto LABEL_19;
  }
  LODWORD(a2) = 0;
LABEL_10:
  v8 = v6 >> 7;
  if ( !(_BYTE)v5 )
    goto LABEL_11;
LABEL_19:
  unk_4F968EA |= 1u;
  if ( v4 != 6 )
  {
    v8 &= ~(unsigned __int64)((v4 == 0) & (unsigned __int8)((_BYTE)v5 == 64));
LABEL_11:
    v9 = (_DWORD)a2 << 23;
    if ( !v8 )
      v9 = 0;
    return (unsigned int)((a1 << 31) + v9 + v8);
  }
  LODWORD(v8) = v8 | 1;
  v9 = (_DWORD)a2 << 23;
  return (unsigned int)((a1 << 31) + v9 + v8);
}

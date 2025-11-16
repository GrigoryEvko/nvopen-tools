// Function: sub_12F9B80
// Address: 0x12f9b80
//
unsigned __int64 __fastcall sub_12F9B80(unsigned __int8 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  char v4; // r14
  int v6; // r13d
  unsigned __int64 v7; // rax
  unsigned __int64 v9; // r12
  int v10; // esi
  bool v11; // di
  unsigned int v12; // r13d

  v3 = 8;
  v4 = unk_4F968EB;
  if ( (unk_4F968EB & 0xFB) != 0 )
  {
    v3 = 15;
    if ( unk_4F968EB != (a1 == 0) + 2 )
      v3 = 0;
  }
  LOBYTE(v6) = a3 & 0xF;
  if ( (unsigned int)a2 <= 0x1C )
  {
    v7 = v3 + a3;
    goto LABEL_10;
  }
  if ( a2 >= 0 )
  {
    if ( a2 != 29 || (v7 = a3 + (unsigned __int8)v3, v7 > 0x7FFF) )
    {
      sub_12F9B70(5);
      return (a1 << 15) - (((_BYTE)v3 == 0) - 31744);
    }
    goto LABEL_10;
  }
  v11 = unk_4C6F00D == 0 || a2 != -1;
  if ( v11 )
  {
    v12 = a3 != 0;
    if ( a2 < -30 )
      goto LABEL_16;
  }
  else
  {
    v11 = (unsigned __int64)(a3 + v3) <= 0x7FFF;
  }
  v12 = ((unsigned int)a3 >> -(char)a2) | ((_DWORD)a3 << a2 != 0);
LABEL_16:
  v7 = v3 + v12;
  v6 = v12 & 0xF;
  if ( v6 && v11 )
  {
    v9 = v7 >> 4;
    sub_12F9B70(2);
    LOWORD(a2) = 0;
    goto LABEL_19;
  }
  LOWORD(a2) = 0;
LABEL_10:
  v9 = v7 >> 4;
  if ( !(_BYTE)v6 )
    goto LABEL_11;
LABEL_19:
  unk_4F968EA |= 1u;
  if ( v4 != 6 )
  {
    v9 &= ~(unsigned __int64)((v4 == 0) & (unsigned __int8)((_BYTE)v6 == 8));
LABEL_11:
    v10 = (unsigned __int16)a2 << 10;
    if ( !v9 )
      v10 = 0;
    return v9 + v10 + (a1 << 15);
  }
  v9 |= 1u;
  v10 = (unsigned __int16)a2 << 10;
  return v9 + v10 + (a1 << 15);
}

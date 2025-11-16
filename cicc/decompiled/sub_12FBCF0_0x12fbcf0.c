// Function: sub_12FBCF0
// Address: 0x12fbcf0
//
__int64 __fastcall sub_12FBCF0(unsigned int a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // r12
  char v6; // r14
  __int64 v7; // r13
  unsigned __int64 v8; // rbx
  unsigned __int64 v10; // rbx
  __int64 v11; // rcx
  bool v12; // si
  bool v13; // si
  unsigned __int64 v14; // rdx

  v3 = a2;
  v4 = a1;
  v5 = 512;
  v6 = unk_4F968EB;
  if ( (unk_4F968EB & 0xFB) != 0 )
  {
    v5 = 1023;
    if ( unk_4F968EB != ((_BYTE)a1 == 0) + 2 )
      v5 = 0;
  }
  v7 = a3 & 0x3FF;
  if ( (unsigned __int16)a2 <= 0x7FCu )
  {
    v8 = v5 + a3;
    goto LABEL_10;
  }
  if ( a2 >= 0 )
  {
    if ( a2 != 2045 || (v8 = v5 + a3, (__int64)(v5 + a3) < 0) )
    {
      sub_12F9B70(5);
      return (__PAIR128__(((unsigned __int64)a1 << 63) + 0x7FF0000000000000LL, v5) - 1) >> 64;
    }
    goto LABEL_10;
  }
  v12 = a2 != -1 || unk_4C6F00D == 0;
  if ( v12 )
  {
    if ( v3 < -62 )
    {
      v13 = a3 != 0;
      v14 = a3 != 0;
      v7 = v14;
      goto LABEL_17;
    }
  }
  else
  {
    v12 = (__int64)(v5 + a3) >= 0;
  }
  v14 = (a3 << v3 != 0) | (a3 >> -(char)v3);
  v7 = v14 & 0x3FF;
  v13 = (v14 & 0x3FF) != 0 && v12;
LABEL_17:
  v8 = v5 + v14;
  v3 = 0;
  if ( v13 )
  {
    v10 = v8 >> 10;
    sub_12F9B70(2);
    v4 = a1;
    v3 = 0;
    goto LABEL_19;
  }
LABEL_10:
  v10 = v8 >> 10;
  if ( !v7 )
    goto LABEL_11;
LABEL_19:
  unk_4F968EA |= 1u;
  if ( v6 != 6 )
  {
    v10 &= ~(unsigned __int64)((v6 == 0) & (unsigned __int8)(v7 == 512));
LABEL_11:
    v11 = v3 << 52;
    if ( !v10 )
      v11 = 0;
    return v10 + v11 + (v4 << 63);
  }
  v10 |= 1u;
  v11 = v3 << 52;
  return v10 + v11 + (v4 << 63);
}

// Function: sub_F51790
// Address: 0xf51790
//
__int64 __fastcall sub_F51790(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r14d
  unsigned __int8 *v5; // r13
  unsigned __int8 v6; // al
  unsigned int v7; // eax
  unsigned int v8; // ebx
  __int64 result; // rax
  __int16 v10; // cx
  unsigned __int64 v11; // rdx
  unsigned int v12; // eax
  unsigned __int64 v13; // rdx

  v4 = a2;
  v5 = sub_BD3990(a1, a2);
  v6 = *v5;
  if ( *v5 > 0x1Cu )
  {
    if ( v6 != 60 )
      return 0;
    v10 = *((_WORD *)v5 + 1);
    _BitScanReverse64(&v11, 1LL << v10);
    LODWORD(v11) = v11 ^ 0x3F;
    result = (unsigned int)(63 - v11);
    if ( (unsigned __int8)(63 - v11) < (unsigned __int8)a2
      && (!*(_BYTE *)(a3 + 17) || *(_BYTE *)(a3 + 16) >= (unsigned __int8)a2) )
    {
      *((_WORD *)v5 + 1) = (unsigned __int8)a2 | v10 & 0xFFC0;
      return (unsigned int)a2;
    }
  }
  else
  {
    if ( (unsigned __int8)(v6 - 2) > 1u && v6 )
      return 0;
    LOBYTE(v7) = sub_BD5420(v5, a3);
    v8 = v7;
    if ( (unsigned __int8)v7 < (unsigned __int8)a2 && (unsigned __int8)sub_B2FCD0((__int64)v5) )
    {
      if ( (v5[33] & 0x1C) != 0 )
      {
        v12 = sub_BAA8B0(*((_QWORD *)v5 + 5));
        if ( v12 > 7 )
        {
          _BitScanReverse64(&v13, v12 >> 3);
          LODWORD(v13) = v13 ^ 0x3F;
          v4 = 63 - v13;
          if ( (unsigned __int8)a2 <= (unsigned __int8)(63 - v13) )
            v4 = a2;
        }
      }
      sub_B2F770((__int64)v5, v4);
      return v4;
    }
    else
    {
      return v8;
    }
  }
  return result;
}

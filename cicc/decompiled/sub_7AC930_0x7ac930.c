// Function: sub_7AC930
// Address: 0x7ac930
//
__int64 __fastcall sub_7AC930(__int64 a1, int a2, __int64 a3, int a4, unsigned int a5)
{
  __int64 v5; // rsi
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 17) & 0x40) == 0 )
  {
    *(_BYTE *)(a1 + 16) &= ~0x80u;
    *(_QWORD *)(a1 + 24) = 0;
  }
  if ( a2 )
  {
    result = sub_7D4600(unk_4F07288, a1, a5);
  }
  else
  {
    if ( !a3 )
      goto LABEL_8;
    v5 = *(_QWORD *)(a3 + 64);
    if ( (*(_BYTE *)(a3 + 81) & 0x10) != 0 )
    {
      result = sub_7D2AC0(a1, v5, a5);
      goto LABEL_10;
    }
    if ( v5 )
    {
      result = sub_7D4A40(a1, v5, a5);
    }
    else
    {
LABEL_8:
      if ( a4 )
        return 0;
      result = sub_7D5DD0(a1, a5);
    }
  }
LABEL_10:
  if ( !result || *(_BYTE *)(result + 80) == 19 )
    return 0;
  return result;
}

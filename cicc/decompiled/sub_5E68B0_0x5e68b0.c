// Function: sub_5E68B0
// Address: 0x5e68b0
//
__int64 __fastcall sub_5E68B0(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 result; // rax
  unsigned __int8 v6; // dl
  char v7; // dl
  char v8; // dl

  v4 = *(_QWORD *)(a1 + 64);
  result = *(_QWORD *)(a1 + 88);
  if ( (*(_BYTE *)(v4 + 179) & 0x40) != 0 )
  {
LABEL_6:
    v8 = *(_BYTE *)(result + 88);
    *(_BYTE *)(result + 172) = 2;
    *(_BYTE *)(result + 88) = v8 & 0x8F | 0x10;
    return result;
  }
  v6 = (*(_BYTE *)(v4 + 88) >> 4) & 7;
  if ( v6 > 1u )
  {
    if ( !a2 || unk_4D04824 )
    {
      v7 = *(_BYTE *)(result + 88);
      *(_BYTE *)(result + 172) = 1;
      *(_BYTE *)(result + 88) = v7 & 0x8F | 0x20;
      return sub_649830(a1, a3, 1);
    }
    goto LABEL_6;
  }
  *(_BYTE *)(result + 88) = *(_BYTE *)(result + 88) & 0x8F | (16 * v6);
  return result;
}

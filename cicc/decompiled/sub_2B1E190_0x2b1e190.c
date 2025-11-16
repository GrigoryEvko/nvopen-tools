// Function: sub_2B1E190
// Address: 0x2b1e190
//
__int64 __fastcall sub_2B1E190(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 result; // rax
  unsigned int v7; // ecx
  int v8; // eax
  int v9; // esi
  unsigned int v10; // eax
  unsigned int v11; // eax
  unsigned int v12; // esi

  v3 = a2;
  v4 = a2;
  if ( (_BYTE)qword_5010508 && *(_BYTE *)(a2 + 8) == 17 )
    v3 = **(_QWORD **)(a2 + 16);
  if ( !(unsigned __int8)sub_BCBCB0(v3) || (*(_BYTE *)(v3 + 8) & 0xFD) == 4 )
    goto LABEL_6;
  v8 = *(unsigned __int8 *)(a2 + 8);
  if ( (_BYTE)v8 == 17 )
  {
    v9 = a3 * *(_DWORD *)(a2 + 32);
LABEL_11:
    v4 = **(_QWORD **)(v4 + 16);
    goto LABEL_12;
  }
  v9 = a3;
  if ( (unsigned int)(v8 - 17) <= 1 )
    goto LABEL_11;
LABEL_12:
  sub_BCDA70((__int64 *)v4, v9);
  v10 = sub_DFDB60(a1);
  if ( v10 && a3 > v10 )
  {
    v11 = (a3 != 0) + (a3 - (a3 != 0)) / v10;
    v12 = 1;
    if ( v11 > 1 )
    {
      _BitScanReverse(&v11, v11 - 1);
      v12 = 1 << (32 - (v11 ^ 0x1F));
    }
    if ( a3 >= v12 )
      return v12 * (a3 / v12);
  }
LABEL_6:
  result = 0;
  if ( a3 )
  {
    _BitScanReverse(&v7, a3);
    return 0x80000000 >> (v7 ^ 0x1F);
  }
  return result;
}

// Function: sub_2598270
// Address: 0x2598270
//
bool __fastcall sub_2598270(__int64 *a1, _BYTE *a2)
{
  __int64 v2; // rdx
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdx
  char v6; // al
  unsigned __int64 v8[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( (unsigned __int8)(*a2 - 34) <= 0x33u )
  {
    v2 = 0x8000000000041LL;
    if ( _bittest64(&v2, (unsigned int)(unsigned __int8)*a2 - 34) )
    {
      v3 = *a1;
      sub_250D230(v8, (unsigned __int64)a2, 5, 0);
      v4 = sub_25294B0(v3, v8[0], v8[1], a1[1], 0, 0, 1);
      if ( v4 )
      {
        v5 = a1[1];
        v6 = *(_BYTE *)(v5 + 97) & *(_BYTE *)(v4 + 97);
LABEL_5:
        *(_BYTE *)(v5 + 97) = *(_BYTE *)(v5 + 96) | v6;
        return *(_BYTE *)(a1[1] + 97) != *(_BYTE *)(a1[1] + 96);
      }
    }
  }
  if ( (unsigned __int8)sub_B46420((__int64)a2) )
    *(_BYTE *)(a1[1] + 97) = *(_BYTE *)(a1[1] + 96) | *(_BYTE *)(a1[1] + 97) & 0xFE;
  if ( (unsigned __int8)sub_B46490((__int64)a2) )
  {
    v5 = a1[1];
    v6 = *(_BYTE *)(v5 + 97) & 0xFD;
    goto LABEL_5;
  }
  return *(_BYTE *)(a1[1] + 97) != *(_BYTE *)(a1[1] + 96);
}

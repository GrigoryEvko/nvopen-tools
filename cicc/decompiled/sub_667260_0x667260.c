// Function: sub_667260
// Address: 0x667260
//
__int64 __fastcall sub_667260(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // r8
  __int64 result; // rax
  __int64 v6; // rax
  int v7; // [rsp+8h] [rbp-18h]
  int v8; // [rsp+Ch] [rbp-14h]

  if ( dword_4F04C64 != -1
    && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) != 0
    && dword_4F077C4 == 2
    && (*(_BYTE *)(a1 - 8) & 1) != 0
    && (*(_BYTE *)(a2 + 18) & 0x40) == 0 )
  {
    v7 = a4;
    v8 = a3;
    v6 = sub_7CAFF0(a2, a1, a3);
    a4 = v7;
    v4 = v6;
    if ( v8 )
    {
      if ( v6 )
        *(_BYTE *)(v6 + 33) |= 0x10u;
      goto LABEL_12;
    }
    return sub_86A320(a1, 0, v4, a4 == 0 ? 16 : 80);
  }
  if ( !(_DWORD)a3 )
  {
    v4 = 0;
    return sub_86A320(a1, 0, v4, a4 == 0 ? 16 : 80);
  }
LABEL_12:
  result = *(_BYTE *)(a1 + 90) & 0xBF;
  *(_BYTE *)(a1 + 90) = *(_BYTE *)(a1 + 90) & 0xBF | ((a4 & 1) << 6);
  return result;
}

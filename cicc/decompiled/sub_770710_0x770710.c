// Function: sub_770710
// Address: 0x770710
//
__int64 __fastcall sub_770710(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v5; // rdi
  int v6; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v7[3]; // [rsp+8h] [rbp-18h] BYREF

  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v5 = *(_QWORD *)(a2 + 16);
      if ( *(_BYTE *)(v5 + 173) == 1 )
      {
        sub_620E00((_WORD *)(v5 + 176), 0, v7, &v6);
        if ( !v6 && !v7[0] )
        {
          *(_OWORD *)a2 = 0;
          *(_OWORD *)(a2 + 16) = 0;
          return 1;
        }
      }
    }
    return 0;
  }
  v3 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)(v3 + 173) != 1 )
    return 0;
  sub_620E00((_WORD *)(v3 + 176), 0, v7, &v6);
  if ( v6 || v7[0] )
    return 0;
  *(_OWORD *)a1 = 0;
  *(_OWORD *)(a1 + 16) = 0;
  return 1;
}

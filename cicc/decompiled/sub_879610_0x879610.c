// Function: sub_879610
// Address: 0x879610
//
__int64 __fastcall sub_879610(char *src)
{
  __int64 v1; // rsi
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rax

  v1 = unk_4D04990;
  if ( unk_4D04990 )
    return sub_879550(src, v1, 0);
  if ( qword_4D049B8 )
  {
    v3 = qword_4D049B8[11];
    if ( v3 )
    {
      v4 = sub_879550(src, qword_4D049B8[11], 0);
      if ( v4 && *(_BYTE *)(v4 + 80) == 19 )
      {
        unk_4D04990 = v3;
        v1 = v3;
      }
      else
      {
        v5 = sub_879550("experimental", v3, 0);
        if ( v5
          && *(_BYTE *)(v5 + 80) == 23
          && (v6 = *(_QWORD *)(v5 + 88), (v7 = sub_879550(src, v6, 0)) != 0)
          && *(_BYTE *)(v7 + 80) == 19 )
        {
          unk_4D04990 = v6;
        }
        else
        {
          v6 = unk_4D04990;
        }
        if ( !v6 )
          return 0;
        v1 = v6;
      }
      return sub_879550(src, v1, 0);
    }
  }
  return 0;
}

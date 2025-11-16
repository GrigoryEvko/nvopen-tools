// Function: sub_641C70
// Address: 0x641c70
//
__int64 __fastcall sub_641C70(unsigned int a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // [rsp+8h] [rbp-38h] BYREF
  __int64 v8; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v9[5]; // [rsp+18h] [rbp-28h] BYREF

  v9[0] = *(_QWORD *)&dword_4F063F8;
  sub_7C9F50(&v7, &v8);
  if ( v7 )
  {
    if ( word_4F06418[0] != 75 )
      goto LABEL_3;
    result = unk_4D04160;
    if ( (unk_4D04160 & 0xFFFFFFFD) == 0 )
    {
      if ( !unk_4D03FA8 )
      {
        v2 = sub_87F870(v7, v8, a1, v9);
        unk_4D04160 = 3;
        unk_4D03FA8 = v2;
        result = (__int64)&dword_4D0415C;
        dword_4D0415C = 0;
        if ( !a1 )
        {
          result = dword_4D0444C;
          if ( dword_4D0444C || !v8 )
          {
            v3 = sub_727790();
            v4 = unk_4D03FA8;
            *(_QWORD *)(v3 + 8) = *(_QWORD *)(unk_4D03FA8 + 48LL);
            *(_QWORD *)(v3 + 16) = *(_QWORD *)(v4 + 48);
            v5 = sub_727740(3);
            v6 = unk_4D03FA8;
            *(_QWORD *)(v3 + 32) = v5;
            *(_QWORD *)(v5 + 8) = *(_QWORD *)(*(_QWORD *)v6 + 8LL);
            *(_BYTE *)(v3 + 40) |= 1u;
            return sub_824FF0(v3);
          }
        }
      }
      return result;
    }
    return sub_684AA0(7, 3 * (unsigned int)((_DWORD)result == 3) + 3064, v9);
  }
  sub_6851C0(3069, v9);
  if ( word_4F06418[0] != 75 )
  {
LABEL_3:
    result = unk_4D04160;
    if ( (unk_4D04160 & 0xFFFFFFFD) == 0 )
      return result;
    return sub_684AA0(7, 3 * (unsigned int)((_DWORD)result == 3) + 3064, v9);
  }
  result = unk_4D04160;
  if ( (unk_4D04160 & 0xFFFFFFFD) != 0 )
    return sub_684AA0(7, 3 * (unsigned int)((_DWORD)result == 3) + 3064, v9);
  return result;
}

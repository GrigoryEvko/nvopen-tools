// Function: sub_86E100
// Address: 0x86e100
//
__int64 __fastcall sub_86E100(__int64 a1, _DWORD *a2)
{
  char v3; // al
  int v4; // edx
  int v5; // r13d
  __int64 v6; // rdi
  __int64 result; // rax
  unsigned __int16 v8; // di
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_BYTE *)(a1 + 40);
  v4 = dword_4F077C4;
  if ( v3 == 13 )
  {
    v8 = 26;
    if ( dword_4F077C4 != 2 )
      goto LABEL_9;
    goto LABEL_12;
  }
  v5 = unk_4D04858;
  if ( !unk_4D04858 )
  {
    v6 = 0;
    if ( dword_4F077C4 != 2 )
      goto LABEL_4;
    v8 = 10;
    goto LABEL_13;
  }
  if ( (unsigned __int8)(v3 - 1) > 1u )
  {
    v8 = 10;
    if ( v3 != 16 )
    {
      if ( dword_4F077C4 != 2 )
      {
LABEL_9:
        v6 = 0;
LABEL_10:
        result = sub_6D7680(v6);
        *(_QWORD *)(a1 + 48) = result;
        return result;
      }
LABEL_12:
      v5 = 0;
      goto LABEL_13;
    }
  }
  v9 = 176LL * unk_4D03B90;
  v11 = v9 + qword_4D03B98;
  v12 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 336);
  *(_BYTE *)(v11 + 5) |= 0x80u;
  v15 = 0;
  *(_QWORD *)(v11 + 144) = v12;
  v10 = v9 + qword_4D03B98;
  *(_QWORD *)(v9 + qword_4D03B98 + 136) = &v15;
  if ( v4 != 2 )
    goto LABEL_18;
  v5 = 1;
  v8 = 10;
LABEL_13:
  if ( (unsigned int)sub_679C10(v8) )
    goto LABEL_21;
  if ( word_4F06418[0] == 75 )
  {
    if ( !v5 )
      goto LABEL_25;
LABEL_21:
    result = sub_86DAC0(a1, 0);
    goto LABEL_22;
  }
  if ( !v5 )
  {
LABEL_25:
    v3 = *(_BYTE *)(a1 + 40);
    v6 = 0;
LABEL_4:
    if ( v3 == 16 )
    {
      result = sub_6B9750(1, v6);
      *(_QWORD *)(a1 + 48) = result;
      return result;
    }
    goto LABEL_10;
  }
  v9 = 176LL * unk_4D03B90;
  v10 = v9 + qword_4D03B98;
LABEL_18:
  *(_BYTE *)(v10 + 5) &= ~0x80u;
  v13 = qword_4D03B98;
  *(_QWORD *)(v10 + 144) = 0;
  *(_QWORD *)(v13 + v9 + 136) = 0;
  v14 = sub_6A3CB0(*(_BYTE *)(a1 + 40) == 2);
  v6 = v14;
  if ( word_4F06418[0] != 75 )
  {
    v3 = *(_BYTE *)(a1 + 40);
    goto LABEL_4;
  }
  result = sub_86DAC0(a1, v14);
LABEL_22:
  *a2 = 1;
  return result;
}

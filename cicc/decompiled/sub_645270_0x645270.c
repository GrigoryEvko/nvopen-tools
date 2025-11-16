// Function: sub_645270
// Address: 0x645270
//
void __fastcall sub_645270(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  unsigned int v6; // r12d
  __int64 v7; // r15
  __int16 v8; // r13
  char v9; // al
  __int64 v10; // rdx
  __int64 v11; // rcx
  _QWORD v12[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( word_4F06418[0] != 149 )
    return;
  v12[0] = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(a1, a2, a3, a4);
  if ( !(unsigned int)sub_7BE280(27, 125, 0, 0) )
    return;
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  if ( word_4F06418[0] != 7 )
  {
    sub_6851D0(1038);
    if ( word_4F06418[0] == 28 )
    {
      sub_7B8B50(1038, 125, v10, v11);
      --*(_BYTE *)(qword_4F061C8 + 36LL);
      return;
    }
LABEL_13:
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    return;
  }
  if ( !unk_4F063AD )
  {
    sub_7B8B50(27, 125, v4, v5);
    sub_7BE280(28, 18, 0, 0);
    goto LABEL_13;
  }
  v6 = dword_4F063F8;
  v7 = unk_4F063B8;
  v8 = unk_4F063FC;
  sub_7B8B50(27, 125, v4, v5);
  sub_7BE280(28, 18, 0, 0);
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  if ( v7 )
  {
    v9 = *(_BYTE *)(a1 + 268);
    if ( v9 == 4 )
    {
      sub_684B30(1117, v12);
    }
    else if ( dword_4F04C58 == -1
           || v9 != 3 && v9
           || (unsigned int)sub_8D2310(*(_QWORD *)(a1 + 288)) && *(char *)(a1 + 125) >= 0 )
    {
      *(_QWORD *)(a1 + 240) = v7;
      *(_DWORD *)(a1 + 248) = v6;
      *(_WORD *)(a1 + 252) = v8;
    }
    else
    {
      sub_684B30(1168, v12);
    }
  }
}

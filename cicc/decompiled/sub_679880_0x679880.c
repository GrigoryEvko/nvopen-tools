// Function: sub_679880
// Address: 0x679880
//
__int64 __fastcall sub_679880(__int64 a1)
{
  _BYTE v2[64]; // [rsp+0h] [rbp-40h] BYREF

  if ( *(_DWORD *)(a1 + 32) )
  {
    sub_7BDC00();
    if ( *(_DWORD *)(unk_4F061C0 + 12LL) != *(_DWORD *)(a1 + 48) || word_4F06418[0] == 9 )
    {
      sub_7ADF70(v2, 0);
      sub_7AE700(unk_4F061C0 + 24LL, *(unsigned int *)(a1 + 48), unk_4F0664C, 1, v2);
      sub_7BBF80(v2, word_4F06418[0] != 9);
    }
  }
  return sub_866A20(
           *(unsigned int *)(a1 + 28),
           *(_QWORD *)(a1 + 56),
           *(unsigned int *)(a1 + 36),
           *(unsigned int *)(a1 + 40));
}

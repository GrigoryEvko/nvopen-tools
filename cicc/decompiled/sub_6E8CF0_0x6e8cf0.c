// Function: sub_6E8CF0
// Address: 0x6e8cf0
//
__int64 __fastcall sub_6E8CF0(__int64 a1)
{
  __int64 v2; // rdi
  _DWORD v4[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = qword_4D03A58;
  if ( !qword_4D03A58 )
  {
    v4[0] = 0;
    if ( dword_4F077C4 == 2 || unk_4F07778 <= 199900 || dword_4F077C0 )
    {
      qword_4D03A58 = sub_724D80(4);
      *(_QWORD *)(qword_4D03A58 + 128) = sub_72C6F0(2);
      sub_70B680(2, 0, *(_QWORD *)(qword_4D03A58 + 176), v4);
      sub_70B680(2, 1, *(_QWORD *)(qword_4D03A58 + 176) + 16LL, v4);
    }
    else
    {
      qword_4D03A58 = sub_724D80(5);
      *(_QWORD *)(qword_4D03A58 + 128) = sub_72C7D0(2);
      sub_70B680(2, 1, qword_4D03A58 + 176, v4);
    }
    v2 = qword_4D03A58;
  }
  return sub_6E6A50(v2, a1);
}

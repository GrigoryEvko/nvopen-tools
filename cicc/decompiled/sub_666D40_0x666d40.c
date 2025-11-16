// Function: sub_666D40
// Address: 0x666d40
//
__int64 __fastcall sub_666D40(__int64 a1, __int16 a2)
{
  unsigned int v2; // r12d
  int v3; // r15d
  int v4; // r14d
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 result; // rax
  __int64 v8; // [rsp-8h] [rbp-68h]
  __int64 v9; // [rsp+8h] [rbp-58h]
  _BYTE v10[80]; // [rsp+10h] [rbp-50h] BYREF

  v2 = dword_4F04C3C;
  dword_4F04C3C = 1;
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) |= 0x10u;
  v3 = unk_4F07798;
  v4 = unk_4F04D84;
  unk_4F07798 = 1;
  if ( a2 )
    sub_721090(a1);
  v9 = unk_4F077C8;
  sub_7ADF70(v10, 0);
  sub_7AE210(v10);
  sub_7BC000(v10);
  sub_7CB300(a1, 0, 0, 0, v9);
  for ( ; word_4F06418[0] != 9; v5 = v8 )
  {
    a1 = 1;
    sub_660E20(1, 0, 1, 0, 0, 0, 0);
  }
  sub_7B8B50(a1, 0, v5, v6);
  unk_4F07798 = v3;
  unk_4F04D84 = v4;
  dword_4F04C3C = v2;
  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *(_BYTE *)(result + 7) = *(_BYTE *)(result + 7) & 0xEF | (16 * (v2 & 1));
  return result;
}

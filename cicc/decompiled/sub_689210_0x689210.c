// Function: sub_689210
// Address: 0x689210
//
__int64 __fastcall sub_689210(__int64 a1, int a2)
{
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 result; // rax

  v3 = *(_QWORD *)(unk_4D03C50 + 48LL);
  if ( a2 )
    sub_6891A0();
  if ( !*(_BYTE *)(a1 + 8) )
  {
    v4 = *(_QWORD *)(a1 + 24) + 8LL;
    if ( v3 )
    {
      v5 = unk_4D03C50;
      *(_QWORD *)(a1 + 32) = *(_QWORD *)(unk_4D03C50 + 48LL);
      v6 = *(_QWORD *)(qword_4F06BC0 + 32LL);
      *(_QWORD *)(v5 + 48) = 0;
      v7 = *(_QWORD *)(a1 + 32);
      qword_4F06BC0 = v6;
      sub_7347F0(v7);
    }
    *(_BYTE *)(a1 + 9) |= 1u;
    sub_6E1850(v4);
  }
  result = unk_4D03C50;
  if ( (*(_BYTE *)(unk_4D03C50 + 20LL) & 4) != 0 )
    *(_BYTE *)(a1 + 9) |= 0x80u;
  return result;
}

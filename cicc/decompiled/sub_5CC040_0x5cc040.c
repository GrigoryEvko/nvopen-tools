// Function: sub_5CC040
// Address: 0x5cc040
//
__int64 __fastcall sub_5CC040(char a1)
{
  int v1; // r15d
  __int16 v2; // r14
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  _QWORD *v7; // rax

  v1 = unk_4F063F8;
  v2 = unk_4F063FC;
  if ( unk_4D04320 )
    sub_684B30(1610, &unk_4F063F8);
  sub_7B8B50();
  sub_7BE280(27, 125, 0, 0);
  sub_7BE280(27, 125, 0, 0);
  ++*(_BYTE *)(unk_4F061C8 + 36LL);
  v3 = sub_5CBEA0(a1, 2, 28, 0);
  sub_7BE280(28, 18, 0, 0);
  if ( v3 )
  {
    v4 = sub_727710();
    *(_DWORD *)v4 = v1;
    v5 = v4;
    *(_WORD *)(v4 + 4) = v2;
    unk_4F061D8 = unk_4F063F0;
    v6 = unk_4F063F8;
    *(_QWORD *)(v4 + 8) = unk_4F063F0;
    unk_4D04178 = v6;
    unk_4D04180 = unk_4F06650;
    v7 = (_QWORD *)v3;
    do
    {
      v7[5] = v5;
      v7 = (_QWORD *)*v7;
    }
    while ( v7 );
  }
  sub_7BE280(28, 18, 0, 0);
  --*(_BYTE *)(unk_4F061C8 + 36LL);
  return v3;
}

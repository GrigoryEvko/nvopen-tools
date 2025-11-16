// Function: sub_600AA0
// Address: 0x600aa0
//
void __fastcall sub_600AA0(__int64 *a1)
{
  __int64 v1; // r13
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // r14
  __int16 v6; // ax
  __int64 v7; // r14
  __int64 v8; // r13
  int v9; // eax
  _BYTE v10[112]; // [rsp+0h] [rbp-70h] BYREF

  v1 = *a1;
  v2 = *(_QWORD *)(*a1 + 96);
  if ( !*(_QWORD *)(v2 + 24) && (*(_BYTE *)(v2 + 177) & 2) != 0 )
  {
    v3 = sub_7259C0(7);
    v4 = *(_QWORD *)(v3 + 168);
    v5 = v3;
    *(_QWORD *)(v3 + 160) = sub_72CBE0();
    v6 = *(_WORD *)(v4 + 16);
    *(_BYTE *)(v4 + 21) |= 1u;
    *(_QWORD *)(v4 + 40) = a1;
    *(_WORD *)(v4 + 16) = v6 & 0x8DFD | 0x2202;
    sub_7325D0(v5, &unk_4F077C8);
    v7 = sub_646F50(v5, 2, 0xFFFFFFFFLL);
    sub_725ED0(v7, 2);
    *(_DWORD *)(v7 + 192) |= 0x81000u;
    sub_736C90(v7, 1);
    sub_878710(v1, v10);
    sub_87A530(v10, 0);
    v8 = sub_87EF90(10, v10);
    v9 = *(_DWORD *)(v2 + 96);
    *(_BYTE *)(v8 + 81) |= 0x10u;
    *(_DWORD *)(v8 + 40) = v9;
    *(_QWORD *)(v8 + 64) = a1;
    *(_QWORD *)(v8 + 88) = v7;
    sub_877D80(v7, v8);
    sub_877E20(v8, v7, a1);
    sub_5E68B0(v8, 1, (__int64)dword_4F07508);
    sub_886160(v8);
    sub_7362F0(v7, 0xFFFFFFFFLL);
    if ( dword_4F068EC )
      sub_89A080(v7);
    *(_QWORD *)(v2 + 24) = v8;
  }
}

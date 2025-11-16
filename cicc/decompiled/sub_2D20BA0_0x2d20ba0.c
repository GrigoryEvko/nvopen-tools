// Function: sub_2D20BA0
// Address: 0x2d20ba0
//
__int64 __fastcall sub_2D20BA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 i; // [rsp+8h] [rbp-48h]
  __int64 v12; // [rsp+10h] [rbp-40h] BYREF
  char v13; // [rsp+18h] [rbp-38h]

  v4 = a3 + 24;
  v5 = sub_BC0510(a4, &unk_4F82418, a3);
  v6 = *(_QWORD *)(v4 + 8);
  v12 = 0;
  v7 = *(_QWORD *)(v5 + 8);
  v13 = 0;
  for ( i = v7; v4 != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    v8 = 0;
    if ( v6 )
      v8 = v6 - 56;
    if ( !sub_B2FC80(v8) )
    {
      v9 = sub_BC1CD0(i, &unk_4F881D0, v8);
      sub_2D1FFA0((__int64)&v12, v8, v9 + 8);
    }
  }
  sub_2D1FAE0((__int64)&v12);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}

// Function: sub_825C40
// Address: 0x825c40
//
void __fastcall sub_825C40(__int64 a1, __int64 a2)
{
  int v2; // r15d
  __int16 v3; // r14
  _QWORD *v4; // rax
  __int64 v5; // r12
  _BYTE *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rax
  _BYTE *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  _QWORD *v15; // [rsp+8h] [rbp-48h]
  __m128i *v16; // [rsp+10h] [rbp-40h]
  void *v17; // [rsp+18h] [rbp-38h]
  __int64 v18; // [rsp+18h] [rbp-38h]
  _QWORD *v19; // [rsp+18h] [rbp-38h]

  v2 = dword_4D03F38[0];
  v3 = dword_4D03F38[1];
  *(_QWORD *)dword_4D03F38 = a2;
  v4 = sub_726B30(1);
  v5 = (__int64)v4;
  if ( v4 )
  {
    *v4 = *(_QWORD *)dword_4D03F38;
    v4[1] = *(_QWORD *)dword_4D03F38;
  }
  v6 = sub_73E870();
  v17 = sub_7F0830(v6);
  v7 = sub_72BA30(5u);
  *(_QWORD *)(v5 + 48) = sub_73DBF0(0x1Du, (__int64)v7, (__int64)v17);
  v8 = sub_726B30(11);
  if ( v8 )
  {
    *v8 = *(_QWORD *)dword_4D03F38;
    v8[1] = *(_QWORD *)dword_4D03F38;
  }
  *(_QWORD *)(v5 + 72) = v8;
  v15 = v8;
  v18 = qword_4F04C50;
  v9 = sub_72BA30(5u);
  v16 = sub_7E7C20((__int64)v9, v18, 0, 0);
  v19 = sub_73A830(0, 5u);
  v10 = sub_731250((__int64)v16);
  v14 = sub_698020(v10, 73, (__int64)v19, v11, v12, v13);
  v15[9] = sub_732B10(v14);
  sub_7E6810(v5, a1, 1);
  dword_4D03F38[0] = v2;
  LOWORD(dword_4D03F38[1]) = v3;
}

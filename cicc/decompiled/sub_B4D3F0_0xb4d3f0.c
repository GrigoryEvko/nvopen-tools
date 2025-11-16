// Function: sub_B4D3F0
// Address: 0xb4d3f0
//
__int64 __fastcall sub_B4D3F0(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  char v10; // al
  __int64 v11; // r9
  __int64 v13; // [rsp-10h] [rbp-50h]
  __int64 v15; // [rsp+8h] [rbp-38h]

  v15 = *(_QWORD *)(a2 + 8);
  v9 = sub_AA4E30(*(_QWORD *)(a5 + 16));
  v10 = sub_AE5020(v9, v15);
  sub_B4D3C0(a1, a2, a3, a4, v10, v11, a5, a6);
  return v13;
}

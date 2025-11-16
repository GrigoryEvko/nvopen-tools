// Function: sub_1AABD30
// Address: 0x1aabd30
//
bool __fastcall sub_1AABD30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD v9[2]; // [rsp+0h] [rbp-40h] BYREF
  int v10; // [rsp+10h] [rbp-30h]

  v2 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9E06C, 1u);
  if ( v2 && (v3 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_4F9E06C)) != 0 )
    v4 = v3 + 160;
  else
    v4 = 0;
  v5 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9920C, 1u);
  if ( v5 && (v6 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v5 + 104LL))(v5, &unk_4F9920C)) != 0 )
    v7 = v6 + 160;
  else
    v7 = 0;
  v9[0] = v4;
  v9[1] = v7;
  v10 = (int)&loc_1000000;
  return (unsigned int)sub_1AA6570(a2, (__int64)v9) != 0;
}

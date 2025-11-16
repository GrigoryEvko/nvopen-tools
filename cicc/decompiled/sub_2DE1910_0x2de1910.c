// Function: sub_2DE1910
// Address: 0x2de1910
//
__int64 __fastcall sub_2DE1910(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  unsigned int v5; // r12d
  int v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v8; // [rsp+8h] [rbp-28h]
  __int64 v9; // [rsp+10h] [rbp-20h]

  v2 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8780C);
  if ( v2 && (v3 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_4F8780C)) != 0 )
    v4 = *(_QWORD *)(v3 + 176);
  else
    v4 = 0;
  v9 = v4;
  v7 = 0;
  v8 = 0;
  v5 = sub_2DE1890((__int64)&v7, a2);
  if ( v8 )
    sub_2DDBD80(v8);
  return v5;
}

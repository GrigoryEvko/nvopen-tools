// Function: sub_F41E80
// Address: 0xf41e80
//
bool __fastcall sub_F41E80(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD v12[4]; // [rsp+0h] [rbp-60h] BYREF
  int v13; // [rsp+20h] [rbp-40h]
  char v14; // [rsp+24h] [rbp-3Ch]

  v2 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8144C);
  if ( v2 && (v3 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_4F8144C)) != 0 )
    v4 = v3 + 176;
  else
    v4 = 0;
  v5 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8FBD4);
  if ( v5 && (v6 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v5 + 104LL))(v5, &unk_4F8FBD4)) != 0 )
    v7 = v6 + 176;
  else
    v7 = 0;
  v8 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F875EC);
  if ( v8 && (v9 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v8 + 104LL))(v8, &unk_4F875EC)) != 0 )
    v10 = v9 + 176;
  else
    v10 = 0;
  v12[0] = v4;
  v12[1] = v7;
  v12[2] = v10;
  v12[3] = 0;
  v13 = 0;
  v14 = 1;
  return (unsigned int)sub_F34EF0(a2, (__int64)v12) != 0;
}

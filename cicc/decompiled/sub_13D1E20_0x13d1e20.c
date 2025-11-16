// Function: sub_13D1E20
// Address: 0x13d1e20
//
__int64 *__fastcall sub_13D1E20(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r14
  __int64 v14; // rax

  v5 = sub_160F9A0(*(_QWORD *)(a2 + 8), &unk_4F9E06C, 1);
  if ( v5 && (v6 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v5 + 104LL))(v5, &unk_4F9E06C)) != 0 )
    v7 = v6 + 160;
  else
    v7 = 0;
  v8 = sub_160F9A0(*(_QWORD *)(a2 + 8), &unk_4F9B6E8, 1);
  if ( v8 && (v9 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v8 + 104LL))(v8, &unk_4F9B6E8)) != 0 )
    v10 = v9 + 360;
  else
    v10 = 0;
  v11 = sub_160F9A0(*(_QWORD *)(a2 + 8), &unk_4F9D764, 1);
  if ( v11 && (v12 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v11 + 104LL))(v11, &unk_4F9D764)) != 0 )
    v13 = sub_14CF090(v12, a3);
  else
    v13 = 0;
  v14 = sub_1632FA0(*(_QWORD *)(a3 + 40));
  a1[1] = v10;
  *a1 = v14;
  a1[2] = v7;
  a1[3] = v13;
  a1[4] = 0;
  return a1;
}

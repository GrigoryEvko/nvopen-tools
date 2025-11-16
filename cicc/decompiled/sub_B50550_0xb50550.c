// Function: sub_B50550
// Address: 0xb50550
//
__int64 __fastcall sub_B50550(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = sub_AD64C0(*(_QWORD *)(a1 + 8), 0, 0);
  v10 = *(_QWORD *)(a1 + 8);
  v7 = sub_BD2C40(72, unk_3F2B48C);
  v8 = v7;
  if ( v7 )
    sub_B503D0(v7, 15, v6, a1, v10, a2, a3, a4);
  return v8;
}

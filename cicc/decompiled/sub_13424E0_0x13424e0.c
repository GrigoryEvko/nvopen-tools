// Function: sub_13424E0
// Address: 0x13424e0
//
bool __fastcall sub_13424E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r10
  __int64 v10; // rax
  __int64 v12; // [rsp+0h] [rbp-240h]
  _QWORD *v13; // [rsp+8h] [rbp-238h]
  _QWORD v14[16]; // [rsp+10h] [rbp-230h] BYREF
  _QWORD v15[54]; // [rsp+90h] [rbp-1B0h] BYREF

  v6 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    v12 = a5;
    sub_130D500(v15);
    v6 = v15;
    a5 = v12;
  }
  v13 = v6;
  memset(v14, 0, sizeof(v14));
  v10 = *(_QWORD *)(a4 + 8);
  v14[2] = a5;
  v14[1] = v10;
  sub_1341260(a1, a2, v6, (__int64)v14, 0, 1, (__int64 *)a3, (unsigned __int64 *)(a3 + 8));
  sub_1341260(a1, a2, v13, a6, 0, 1, (__int64 *)(a3 + 16), (unsigned __int64 *)(a3 + 24));
  return !*(_QWORD *)a3 || !*(_QWORD *)(a3 + 8) || !*(_QWORD *)(a3 + 16) || *(_QWORD *)(a3 + 24) == 0;
}

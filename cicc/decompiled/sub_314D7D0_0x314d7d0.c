// Function: sub_314D7D0
// Address: 0x314d7d0
//
__int64 __fastcall sub_314D7D0(unsigned __int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // rbx
  unsigned __int64 v13; // [rsp+8h] [rbp-378h] BYREF
  __int64 v14[110]; // [rsp+10h] [rbp-370h] BYREF

  sub_2353240((__int64)v14, a2, a3, a4, a5, a6);
  v6 = (_QWORD *)sub_22077B0(0x358u);
  v11 = (unsigned __int64)v6;
  if ( v6 )
  {
    *v6 = &unk_4A11938;
    sub_2353240((__int64)(v6 + 1), v14, v7, v8, v9, v10);
  }
  v13 = v11;
  sub_314D790(a1, &v13);
  if ( v13 )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v13 + 8LL))(v13);
  return sub_2341D90((__int64)v14);
}

// Function: sub_1076B70
// Address: 0x1076b70
//
__int64 __fastcall sub_1076B70(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rbx
  __int64 v3; // r15
  __int64 **v4; // rbx
  __int64 **v5; // r13
  __int64 *v6; // rdx
  _QWORD *v7; // rbx
  __int64 v9; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD **)(a1 + 104);
  v9 = (*(__int64 (__fastcall **)(_QWORD *))(*v2 + 80LL))(v2);
  v3 = v2[4] - v2[2];
  sub_1076AA0(a1);
  v4 = *(__int64 ***)(a2 + 40);
  v5 = &v4[*(unsigned int *)(a2 + 48)];
  while ( v5 != v4 )
  {
    v6 = *v4++;
    sub_E5CCC0((__int64 *)a2, *(_QWORD **)(a1 + 104), v6);
  }
  v7 = *(_QWORD **)(a1 + 104);
  return (*(__int64 (__fastcall **)(_QWORD *))(*v7 + 80LL))(v7) + v7[4] - v7[2] - v3 - v9;
}

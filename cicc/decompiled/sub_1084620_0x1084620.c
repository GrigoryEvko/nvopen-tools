// Function: sub_1084620
// Address: 0x1084620
//
__int64 __fastcall sub_1084620(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  bool v3; // zf
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v7; // [rsp+0h] [rbp-40h] BYREF
  int v8; // [rsp+8h] [rbp-38h]
  __int64 v9; // [rsp+10h] [rbp-30h] BYREF
  int v10; // [rsp+18h] [rbp-28h]

  v2 = *(_QWORD *)(a1 + 1096);
  v3 = *(_BYTE *)(a1 + 1092) == 0;
  v8 = 1;
  v7 = v2;
  *(_QWORD *)(a1 + 104) = &v7;
  if ( v3 )
    return sub_1081400(a1, a2, 0);
  v4 = sub_1081400(a1, a2, 1);
  v5 = *(_QWORD *)(a1 + 1104);
  v10 = 1;
  v9 = v5;
  *(_QWORD *)(a1 + 104) = &v9;
  return v4 + sub_1081400(a1, a2, 2);
}

// Function: sub_1E85CD0
// Address: 0x1e85cd0
//
__int64 __fastcall sub_1E85CD0(int a1)
{
  void *v1; // rax
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  int v7; // r9d
  _BYTE *v8; // rax
  __int64 result; // rax
  int v10[4]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v11)(int *, int *, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v12)(int *, __int64, __int64, __int64, __int64, int); // [rsp+18h] [rbp-28h]

  v1 = sub_16E8CB0();
  v2 = sub_1263B40((__int64)v1, "- lanemask:    ");
  v10[0] = a1;
  v3 = v2;
  v11 = (__int64 (__fastcall *)(int *, int *, int))sub_1DB3470;
  v12 = sub_1DB3430;
  sub_1DB3430(v10, v2, v4, v5, v6, v7);
  v8 = *(_BYTE **)(v3 + 24);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v3 + 16) )
  {
    sub_16E7DE0(v3, 10);
  }
  else
  {
    *(_QWORD *)(v3 + 24) = v8 + 1;
    *v8 = 10;
  }
  result = (__int64)v11;
  if ( v11 )
    return v11(v10, v10, 3);
  return result;
}

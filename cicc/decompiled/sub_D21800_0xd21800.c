// Function: sub_D21800
// Address: 0xd21800
//
__int64 __fastcall sub_D21800(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  _QWORD *v5; // r14
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  _QWORD v10[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v11)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v12)(__int64, __int64); // [rsp+18h] [rbp-28h]

  v2 = *(__int64 **)(a1 + 8);
  v12 = sub_D1A600;
  v11 = sub_D1A5D0;
  v10[0] = a1;
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_12:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F86A88 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_12;
  }
  v5 = *(_QWORD **)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(
                      *(_QWORD *)(v3 + 8),
                      &unk_4F86A88)
                  + 176);
  v6 = sub_22077B0(360);
  v7 = v6;
  if ( v6 )
    sub_D216B0(v6, a2, (__int64)v10, v5);
  v8 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v7;
  if ( v8 )
  {
    sub_D1D5E0(v8);
    j_j___libc_free_0(v8, 360);
  }
  if ( v11 )
    v11(v10, v10, 3);
  return 0;
}

// Function: sub_2F255B0
// Address: 0x2f255b0
//
__int64 __fastcall sub_2F255B0(__int64 a1, __int64 *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  char *v7; // [rsp+0h] [rbp-90h] BYREF
  size_t v8; // [rsp+8h] [rbp-88h]
  _BYTE v9[16]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v10[14]; // [rsp+20h] [rbp-70h] BYREF

  v10[5] = 0x100000000LL;
  v10[6] = &v7;
  v7 = v9;
  v10[0] = &unk_49DD210;
  v8 = 0;
  v9[0] = 0;
  memset(&v10[1], 0, 32);
  sub_CB5980((__int64)v10, 0, 0, 0);
  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_8:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_50208C0 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_8;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_50208C0);
  sub_2F25120((__int64)v10, v5 + 176, a2);
  sub_2241490((unsigned __int64 *)(a1 + 208), v7, v8);
  v10[0] = &unk_49DD210;
  sub_CB5840((__int64)v10);
  if ( v7 != v9 )
    j_j___libc_free_0((unsigned __int64)v7);
  return 0;
}

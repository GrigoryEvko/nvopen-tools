// Function: sub_17D0000
// Address: 0x17d0000
//
_QWORD *__fastcall sub_17D0000(__int64 a1, __int64 a2)
{
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // r14
  __int64 v8; // r12
  __int64 *v9; // rax
  _QWORD *v10; // r12
  __int64 **v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 *v16; // rax
  _QWORD *result; // rax
  __int64 v18[3]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD *v19; // [rsp+18h] [rbp-58h]

  sub_17CE510((__int64)v18, a2, 0, 0, 0);
  v5 = *(unsigned int *)(a1 + 56);
  if ( (unsigned int)v5 >= *(_DWORD *)(a1 + 60) )
  {
    sub_16CD150(a1 + 48, (const void *)(a1 + 64), 0, 8, v3, v4);
    v5 = *(unsigned int *)(a1 + 56);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v5) = a2;
  v6 = v19;
  ++*(_DWORD *)(a1 + 56);
  v7 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v8 = *(_QWORD *)(a1 + 24);
  v9 = (__int64 *)sub_1643330(v6);
  v10 = (_QWORD *)sub_17CFB40(v8, v7, v18, v9, 8u);
  v11 = (__int64 **)sub_1643330(v19);
  v14 = sub_15A06D0(v11, v7, v12, v13);
  v15 = sub_1643360(v19);
  v16 = (__int64 *)sub_159C470(v15, 8, 0);
  result = sub_15E7280(v18, v10, v14, v16, 8u, 0, 0, 0, 0);
  if ( v18[0] )
    return (_QWORD *)sub_161E7C0((__int64)v18, v18[0]);
  return result;
}

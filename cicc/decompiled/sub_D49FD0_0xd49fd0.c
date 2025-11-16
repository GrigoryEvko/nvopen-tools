// Function: sub_D49FD0
// Address: 0xd49fd0
//
__int64 __fastcall sub_D49FD0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  unsigned __int8 *v8; // rax
  size_t v9; // rdx
  void *v10; // rdi
  size_t v12; // [rsp+8h] [rbp-38h]

  v6 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v7 = sub_904010(*a2, "Loop info for function '");
  v8 = (unsigned __int8 *)sub_BD5D20(a3);
  v10 = *(void **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v10 < v9 )
  {
    v7 = sub_CB6200(v7, v8, v9);
  }
  else if ( v9 )
  {
    v12 = v9;
    memcpy(v10, v8, v9);
    *(_QWORD *)(v7 + 32) += v12;
  }
  sub_904010(v7, "':\n");
  sub_D49F60(v6 + 8, *a2);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  sub_AE6EC0(a1, (__int64)&unk_4F82400);
  return a1;
}

// Function: sub_B1B190
// Address: 0xb1b190
//
__int64 __fastcall sub_B1B190(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r13
  const void *v8; // rax
  size_t v9; // rdx
  void *v10; // rdi
  __int64 v11; // rax
  size_t v13; // [rsp+8h] [rbp-38h]

  v7 = sub_904010(*a2, "DominatorTree for function: ");
  v8 = (const void *)sub_BD5D20(a3);
  v10 = *(void **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v10 < v9 )
  {
    v7 = sub_CB6200(v7, v8, v9);
  }
  else if ( v9 )
  {
    v13 = v9;
    memcpy(v10, v8, v9);
    *(_QWORD *)(v7 + 32) += v13;
  }
  sub_904010(v7, "\n");
  v11 = sub_BC1CD0(a4, &unk_4F81450);
  sub_B1B090(v11 + 8, *a2);
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

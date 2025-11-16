// Function: sub_1F203D0
// Address: 0x1f203d0
//
void __fastcall sub_1F203D0(__int64 a1, __int64 a2)
{
  __int64 v4; // rcx
  int v5; // r9d
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 *v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // r14
  __int64 v11; // rbx
  unsigned int v12; // esi
  __int64 *v13; // rax
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // rax
  int v16; // r9d
  __int64 v17; // r14
  int v18; // r9d
  __int64 v19; // [rsp+18h] [rbp-38h] BYREF

  sub_1F15650(a1);
  v6 = *(_QWORD *)a1;
  v7 = *(_QWORD *)(*(_QWORD *)(**(_QWORD **)a1 + 96LL) + 8LL * *(unsigned int *)(*(_QWORD *)a2 + 48LL));
  v8 = (__int64 *)(*(_QWORD *)(*(_QWORD *)a1 + 56LL) + 16LL * *(unsigned int *)(v7 + 48));
  v9 = *v8;
  if ( (*v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v8[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    v9 = sub_1F13A50(
           (_QWORD *)(v6 + 48),
           *(_QWORD *)(v6 + 40),
           *(_QWORD *)(*(_QWORD *)(**(_QWORD **)a1 + 96LL) + 8LL * *(unsigned int *)(*(_QWORD *)a2 + 48LL)),
           v4,
           v7,
           v5);
  v19 = v9;
  v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  v11 = (v9 >> 1) & 3;
  v12 = v11 | *(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  v13 = &v19;
  if ( v12 >= (*(_DWORD *)((*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
             | (unsigned int)(*(__int64 *)(a2 + 8) >> 1) & 3) )
    v13 = (__int64 *)(a2 + 8);
  v14 = sub_1F1B1B0(a1, *v13);
  if ( *(_BYTE *)(a2 + 33)
    && (*(_DWORD *)((*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
      | (unsigned int)((*(__int64 *)(a2 + 16) >> 1) & 3)) >= (*(_DWORD *)(v10 + 24) | (unsigned int)v11) )
  {
    v17 = sub_1F1B330((_QWORD *)a1, v19);
    sub_1F1FA40(a1 + 200, v14, v17, *(unsigned int *)(a1 + 80), a1 + 200, v18);
    sub_1F20330(a1, v17, *(_QWORD *)(a2 + 16));
  }
  else
  {
    v15 = sub_1F1BC20(a1, *(_QWORD *)(a2 + 16));
    sub_1F1FA40(a1 + 200, v14, v15, *(unsigned int *)(a1 + 80), a1 + 200, v16);
  }
}

// Function: sub_2FBE000
// Address: 0x2fbe000
//
void __fastcall sub_2FBE000(__int64 a1, __int64 a2)
{
  __int64 v4; // rcx
  __int64 v5; // rdi
  __int64 *v6; // rdx
  __int64 v7; // rax
  unsigned __int64 v8; // r14
  __int64 v9; // rbx
  unsigned int v10; // esi
  __int64 *v11; // rax
  unsigned __int64 v12; // r15
  unsigned __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // r14
  __int64 v16; // r9
  __int64 v17; // [rsp+18h] [rbp-38h] BYREF

  sub_2FB2500(a1);
  v5 = *(_QWORD *)a1;
  v6 = (__int64 *)(*(_QWORD *)(*(_QWORD *)a1 + 56LL) + 16LL * *(unsigned int *)(*(_QWORD *)a2 + 24LL));
  v7 = *v6;
  if ( (*v6 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v6[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    v7 = sub_2FB0650((_QWORD *)(v5 + 48), *(_QWORD *)(v5 + 40), *(_QWORD *)a2, v4, *(_QWORD *)a2);
  v17 = v7;
  v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
  v9 = (v7 >> 1) & 3;
  v10 = v9 | *(_DWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  v11 = &v17;
  if ( v10 >= (*(_DWORD *)((*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
             | (unsigned int)(*(__int64 *)(a2 + 8) >> 1) & 3) )
    v11 = (__int64 *)(a2 + 8);
  v12 = sub_2FBA5C0(a1, *v11);
  if ( *(_BYTE *)(a2 + 33)
    && (*(_DWORD *)((*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
      | (unsigned int)((*(__int64 *)(a2 + 16) >> 1) & 3)) >= (*(_DWORD *)(v8 + 24) | (unsigned int)v9) )
  {
    v15 = sub_2FBA8B0((__int64 *)a1, v17);
    sub_2FBD6E0(a1 + 192, v12, v15, *(unsigned int *)(a1 + 80), a1 + 192, v16);
    sub_2FBD940(a1, v15, *(_QWORD *)(a2 + 16));
  }
  else
  {
    v13 = sub_2FBA740(a1, *(_QWORD *)(a2 + 16));
    sub_2FBD6E0(a1 + 192, v12, v13, *(unsigned int *)(a1 + 80), a1 + 192, v14);
  }
}

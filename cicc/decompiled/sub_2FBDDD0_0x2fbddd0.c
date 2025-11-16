// Function: sub_2FBDDD0
// Address: 0x2fbddd0
//
__int64 __fastcall sub_2FBDDD0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // r13
  signed __int64 v5; // r12
  __int64 v6; // r15
  __int64 *v7; // rdx
  __int64 v8; // r15
  unsigned int v9; // esi
  int *v11; // r10
  unsigned __int64 v12; // r8
  __int64 *v13; // rdi
  __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  __int64 v16; // r15
  __int64 v17; // r12
  __int64 *v18; // rsi
  __int64 *v19; // rax
  unsigned __int64 v20; // r12
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  signed __int64 v24; // [rsp+0h] [rbp-40h]
  unsigned __int64 v25; // [rsp+8h] [rbp-38h]
  int *v26; // [rsp+8h] [rbp-38h]
  int *v27; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[1] + 32LL) + 152LL) + 16LL * *(unsigned int *)(a2 + 24) + 8);
  if ( ((v4 >> 1) & 3) != 0 )
    v5 = v4 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v4 >> 1) & 3) - 1));
  else
    v5 = *(_QWORD *)(v4 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
  v6 = *(_QWORD *)(a1[9] + 8LL);
  v7 = (__int64 *)sub_2E09D00((__int64 *)v6, v5);
  if ( v7 == (__int64 *)(*(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8)) )
    return v4;
  v8 = (v5 >> 1) & 3;
  v9 = v8 | *(_DWORD *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  if ( (*(_DWORD *)((*v7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v7 >> 1) & 3) > v9 )
    return v4;
  v11 = (int *)v7[2];
  if ( !v11 )
    return v4;
  v12 = *a1;
  v13 = (__int64 *)(*(_QWORD *)(*a1 + 56LL) + 16LL * *(unsigned int *)(a2 + 24));
  v14 = *v13;
  if ( (*v13 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v13[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v27 = v11;
    v23 = sub_2FB0650((_QWORD *)(v12 + 48), *(_QWORD *)(v12 + 40), a2, v5 & 0xFFFFFFFFFFFFFFF8LL, v12);
    v11 = v27;
    v14 = v23;
    v9 = *(_DWORD *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 24) | v8;
  }
  v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
  v16 = (v14 >> 1) & 3;
  v25 = v14 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 <= ((unsigned int)v16 | *(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
    goto LABEL_14;
  v24 = v14;
  v17 = *(_QWORD *)(a1[9] + 8LL);
  v18 = (__int64 *)sub_2E09D00((__int64 *)v17, v14);
  if ( v18 == (__int64 *)(*(_QWORD *)v17 + 24LL * *(unsigned int *)(v17 + 8)) )
    return v4;
  v15 = v25;
  if ( (*(_DWORD *)((*v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v18 >> 1) & 3) > (*(_DWORD *)(v25 + 24)
                                                                                          | (unsigned int)v16) )
    return v4;
  v11 = (int *)v18[2];
  if ( !v11 )
    return v4;
  v5 = v24;
LABEL_14:
  v26 = v11;
  v19 = (__int64 *)sub_2FB0DC0((_QWORD *)(*a1 + 48LL), *(_QWORD *)(*a1 + 40LL), a2, v15, v12);
  v20 = sub_2FB9FE0(a1, *((_DWORD *)a1 + 20), v26, v5, a2, v19);
  sub_2FBD6E0((__int64)(a1 + 24), *(_QWORD *)(v20 + 8), v4, *((unsigned int *)a1 + 20), v21, v22);
  return *(_QWORD *)(v20 + 8);
}

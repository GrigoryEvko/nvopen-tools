// Function: sub_2FBDBC0
// Address: 0x2fbdbc0
//
void __fastcall sub_2FBDBC0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  unsigned __int64 v10; // r8
  __int64 *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 *v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned int v20; // ecx
  __int64 *v21; // r12
  unsigned __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r9
  unsigned __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rdi
  __int64 *v30; // r12
  unsigned __int64 v31; // rax
  __int64 v32; // [rsp+10h] [rbp-40h] BYREF
  __int64 v33; // [rsp+18h] [rbp-38h] BYREF

  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_QWORD *)a2;
  v11 = *(__int64 **)a1;
  v12 = 16LL * *(unsigned int *)(*(_QWORD *)a2 + 24LL);
  v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v9 + 32) + 152LL) + v12 + 8);
  v14 = (__int64 *)(v11[7] + v12);
  v15 = *v14;
  if ( (*v14 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v14[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    v15 = sub_2FB0650(v11 + 6, v11[5], v10, a4, v10);
  v32 = v15;
  v16 = a4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_BYTE *)(a2 + 32) )
  {
    if ( !v16 )
      goto LABEL_12;
    v17 = *(_DWORD *)(v16 + 24) | (a4 >> 1) & 3;
    v20 = *(_DWORD *)((*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24);
  }
  else if ( !v16
         || (v17 = *(_DWORD *)(v16 + 24) | (a4 >> 1) & 3,
             v18 = *(_QWORD *)(a2 + 8),
             v19 = v18 >> 1,
             v20 = *(_DWORD *)((v18 & 0xFFFFFFFFFFFFFFF8LL) + 24),
             v17 <= (unsigned __int64)(v20 | v19 & 3)) )
  {
    *(_DWORD *)(a1 + 80) = a3;
    v29 = a1 + 192;
    v28 = *(_QWORD *)(a2 + 8);
    v26 = a3;
LABEL_15:
    v27 = v13;
    goto LABEL_16;
  }
  if ( v17 < v20 )
  {
LABEL_12:
    *(_DWORD *)(a1 + 80) = a3;
    v30 = (__int64 *)(a2 + 8);
    if ( (*(_DWORD *)((*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)(a2 + 8) >> 1) & 3) >= (*(_DWORD *)((v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v32 >> 1) & 3) )
      v30 = &v32;
    v31 = sub_2FBA5C0(a1, *v30);
    v26 = *(unsigned int *)(a1 + 80);
    v29 = a1 + 192;
    v28 = v31;
    goto LABEL_15;
  }
  *(_DWORD *)(a1 + 80) = a3;
  v21 = (__int64 *)(a2 + 8);
  v22 = sub_2FBA660(a1, a4);
  v23 = *(unsigned int *)(a1 + 80);
  v33 = v22;
  sub_2FBD6E0(a1 + 192, v22, v13, v23, a1 + 192, v24);
  sub_2FB2500(a1);
  if ( (*(_DWORD *)((*v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v21 >> 1) & 3) >= (*(_DWORD *)((v33 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                           | (unsigned int)(v33 >> 1)
                                                                                           & 3) )
    v21 = &v33;
  v25 = sub_2FBA5C0(a1, *v21);
  v10 = a1 + 192;
  v26 = *(unsigned int *)(a1 + 80);
  v27 = v33;
  v28 = v25;
  v29 = a1 + 192;
LABEL_16:
  sub_2FBD6E0(v29, v28, v27, v26, v10, a6);
}

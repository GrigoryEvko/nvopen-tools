// Function: sub_1F1FDE0
// Address: 0x1f1fde0
//
void __fastcall sub_1F1FDE0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v10; // rax
  __int64 *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // r12
  __int64 v14; // r8
  __int64 *v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  unsigned int v18; // edx
  __int64 v19; // rsi
  __int64 v20; // rcx
  unsigned int v21; // esi
  unsigned __int64 v22; // rax
  __int64 v23; // rcx
  int v24; // r9d
  __int64 *v25; // rsi
  unsigned __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 *v31; // rsi
  unsigned __int64 v32; // rax
  __int64 v33; // [rsp+10h] [rbp-40h] BYREF
  __int64 v34; // [rsp+18h] [rbp-38h] BYREF

  v10 = *(_QWORD *)(a1 + 16);
  v11 = *(__int64 **)a1;
  v12 = *(_QWORD *)(v10 + 272);
  v13 = *(_QWORD *)(*(_QWORD *)(v12 + 392) + 16LL * *(unsigned int *)(*(_QWORD *)a2 + 48LL) + 8);
  v14 = *(_QWORD *)(*(_QWORD *)(*v11 + 96) + 8LL * *(unsigned int *)(*(_QWORD *)a2 + 48LL));
  v15 = (__int64 *)(v11[7] + 16LL * *(unsigned int *)(v14 + 48));
  v16 = *v15;
  if ( (*v15 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v15[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    v16 = sub_1F13A50(v11 + 6, v11[5], v14, v12, v14, a6);
  v33 = v16;
  v17 = a4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_BYTE *)(a2 + 32) )
  {
    if ( !v17 )
      goto LABEL_12;
    v18 = *(_DWORD *)(v17 + 24) | (a4 >> 1) & 3;
    v21 = *(_DWORD *)((*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24);
  }
  else if ( !v17
         || (v18 = *(_DWORD *)(v17 + 24) | (a4 >> 1) & 3,
             v19 = *(_QWORD *)(a2 + 8),
             v20 = v19 >> 1,
             v21 = *(_DWORD *)((v19 & 0xFFFFFFFFFFFFFFF8LL) + 24),
             v18 <= (unsigned __int64)(v21 | v20 & 3)) )
  {
    *(_DWORD *)(a1 + 80) = a3;
    v30 = a1 + 200;
    v29 = *(_QWORD *)(a2 + 8);
    v27 = a3;
LABEL_15:
    v28 = v13;
    goto LABEL_16;
  }
  if ( v21 > v18 )
  {
LABEL_12:
    *(_DWORD *)(a1 + 80) = a3;
    v31 = (__int64 *)(a2 + 8);
    if ( (*(_DWORD *)((*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)(a2 + 8) >> 1) & 3) >= (*(_DWORD *)((v16 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v16 >> 1) & 3) )
      v31 = &v33;
    v32 = sub_1F1B1B0(a1, *v31);
    v27 = *(unsigned int *)(a1 + 80);
    v30 = a1 + 200;
    v29 = v32;
    goto LABEL_15;
  }
  *(_DWORD *)(a1 + 80) = a3;
  v22 = sub_1F1B250(a1, a4);
  v23 = *(unsigned int *)(a1 + 80);
  v34 = v22;
  sub_1F1FA40(a1 + 200, v22, v13, v23, a1 + 200, v24);
  sub_1F15650(a1);
  v25 = (__int64 *)(a2 + 8);
  if ( (*(_DWORD *)((*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)(a2 + 8) >> 1) & 3) >= (*(_DWORD *)((v34 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v34 >> 1) & 3) )
    v25 = &v34;
  v26 = sub_1F1B1B0(a1, *v25);
  v14 = a1 + 200;
  v27 = *(unsigned int *)(a1 + 80);
  v28 = v34;
  v29 = v26;
  v30 = a1 + 200;
LABEL_16:
  sub_1F1FA40(v30, v29, v28, v27, v14, a6);
}

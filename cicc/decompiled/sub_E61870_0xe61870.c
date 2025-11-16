// Function: sub_E61870
// Address: 0xe61870
//
unsigned __int64 __fastcall sub_E61870(_QWORD *a1, __int64 a2, const void *a3, __int64 a4, const void *a5, size_t a6)
{
  size_t v6; // r15
  _QWORD *v9; // rdi
  __int64 v10; // rax
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // r13
  signed __int64 v13; // rbx
  __int64 v14; // r8
  void *v15; // rdi
  __int64 v16; // r9
  void *v17; // rdi
  __int64 v18; // rdx
  int v20; // [rsp+8h] [rbp-48h]

  v6 = a6;
  v9 = (_QWORD *)*a1;
  v10 = v9[36];
  v9[46] += 312LL;
  v11 = (v10 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9[37] >= v11 + 312 && v10 )
  {
    v9[36] = v11 + 312;
    v12 = (v10 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  }
  else
  {
    v12 = sub_9D1E70((__int64)(v9 + 36), 312, 312, 3);
  }
  v13 = 16 * a4;
  sub_E81B30(v12, 12, 0);
  v15 = (void *)(v12 + 224);
  *(_QWORD *)(v12 + 40) = v12 + 64;
  v16 = v13 >> 4;
  *(_QWORD *)(v12 + 96) = v12 + 112;
  *(_QWORD *)(v12 + 104) = 0x400000000LL;
  *(_BYTE *)(v12 + 30) = 0;
  *(_QWORD *)(v12 + 32) = 0;
  *(_QWORD *)(v12 + 48) = 0;
  *(_QWORD *)(v12 + 56) = 32;
  *(_QWORD *)(v12 + 208) = v12 + 224;
  *(_QWORD *)(v12 + 216) = 0x200000000LL;
  if ( (unsigned __int64)v13 > 0x20 )
  {
    sub_C8D5F0(v12 + 208, (const void *)(v12 + 224), v13 >> 4, 0x10u, v14, v16);
    v16 = v13 >> 4;
    v15 = (void *)(*(_QWORD *)(v12 + 208) + 16LL * *(unsigned int *)(v12 + 216));
  }
  else if ( !v13 )
  {
    goto LABEL_6;
  }
  v20 = v16;
  memcpy(v15, a3, v13);
  LODWORD(v13) = *(_DWORD *)(v12 + 216);
  LODWORD(v16) = v20;
LABEL_6:
  v17 = (void *)(v12 + 280);
  *(_QWORD *)(v12 + 264) = 0;
  *(_DWORD *)(v12 + 216) = v16 + v13;
  *(_QWORD *)(v12 + 256) = v12 + 280;
  *(_QWORD *)(v12 + 272) = 32;
  if ( a6 > 0x20 )
  {
    sub_C8D290(v12 + 256, (const void *)(v12 + 280), a6, 1u, v14, v12 + 256);
    v17 = (void *)(*(_QWORD *)(v12 + 256) + *(_QWORD *)(v12 + 264));
    goto LABEL_10;
  }
  if ( a6 )
  {
LABEL_10:
    memcpy(v17, a5, a6);
    v6 = a6 + *(_QWORD *)(v12 + 264);
  }
  *(_QWORD *)(v12 + 264) = v6;
  v18 = *(_QWORD *)(*(_QWORD *)(a2 + 288) + 8LL);
  *(_QWORD *)(v12 + 8) = v18;
  *(_DWORD *)(v12 + 24) = *(_DWORD *)(*(_QWORD *)(a2 + 288) + 24LL) + 1;
  **(_QWORD **)(a2 + 288) = v12;
  *(_QWORD *)(a2 + 288) = v12;
  *(_QWORD *)(*(_QWORD *)(v18 + 8) + 8LL) = v12;
  return v12;
}

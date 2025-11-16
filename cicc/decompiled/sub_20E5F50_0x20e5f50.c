// Function: sub_20E5F50
// Address: 0x20e5f50
//
void __fastcall sub_20E5F50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 (*v5)(void); // rdx
  __int64 v6; // rax
  __int64 (*v7)(); // rax
  __int64 v8; // rax
  char *v9; // r13
  __int64 v10; // r12
  size_t v11; // r12
  char *v12; // rax
  char *v13; // r13
  __int64 v14; // rdx
  size_t v15; // r12
  char *v16; // rax
  __int64 v17; // rdx
  char *v18; // r13
  size_t v19; // r12
  char *v20; // rax
  int v21; // r12d
  unsigned int v22; // r12d
  void *v23; // r13
  __int64 v24; // rax

  *(_QWORD *)a1 = off_4985A00;
  v4 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = v4;
  v5 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v6 = 0;
  if ( v5 != sub_1D00B00 )
  {
    v6 = v5();
    a2 = *(_QWORD *)(a1 + 8);
  }
  *(_QWORD *)(a1 + 24) = v6;
  v7 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 112LL);
  if ( v7 == sub_1D00B10 )
  {
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = a3;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = 0;
    *(_DWORD *)(a1 + 64) = 0;
    BUG();
  }
  v8 = v7();
  *(_QWORD *)(a1 + 40) = a3;
  v9 = 0;
  *(_QWORD *)(a1 + 32) = v8;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  v10 = *(unsigned int *)(v8 + 16);
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  if ( v10 )
  {
    v11 = 8 * v10;
    v12 = (char *)sub_22077B0(v11);
    v9 = &v12[v11];
    *(_QWORD *)(a1 + 72) = v12;
    *(_QWORD *)(a1 + 88) = &v12[v11];
    memset(v12, 0, v11);
    v8 = *(_QWORD *)(a1 + 32);
  }
  *(_QWORD *)(a1 + 80) = v9;
  v13 = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = a1 + 104;
  *(_QWORD *)(a1 + 128) = a1 + 104;
  *(_QWORD *)(a1 + 136) = 0;
  v14 = *(unsigned int *)(v8 + 16);
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  if ( v14 )
  {
    v15 = 4 * v14;
    v16 = (char *)sub_22077B0(4 * v14);
    v13 = &v16[v15];
    *(_QWORD *)(a1 + 144) = v16;
    *(_QWORD *)(a1 + 160) = &v16[v15];
    memset(v16, 0, v15);
    v8 = *(_QWORD *)(a1 + 32);
  }
  *(_QWORD *)(a1 + 152) = v13;
  v17 = *(unsigned int *)(v8 + 16);
  v18 = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  if ( v17 )
  {
    v19 = 4 * v17;
    v20 = (char *)sub_22077B0(4 * v17);
    v18 = &v20[v19];
    *(_QWORD *)(a1 + 168) = v20;
    *(_QWORD *)(a1 + 184) = &v20[v19];
    memset(v20, 0, v19);
    v8 = *(_QWORD *)(a1 + 32);
  }
  *(_QWORD *)(a1 + 176) = v18;
  v21 = *(_DWORD *)(v8 + 16);
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 208) = v21;
  *(_QWORD *)(a1 + 200) = 0;
  v22 = (unsigned int)(v21 + 63) >> 6;
  v23 = (void *)malloc(8LL * v22);
  if ( !v23 )
  {
    if ( 8LL * v22 || (v24 = malloc(1u)) == 0 )
      sub_16BD1C0("Allocation failed", 1u);
    else
      v23 = (void *)v24;
  }
  *(_QWORD *)(a1 + 192) = v23;
  *(_QWORD *)(a1 + 200) = v22;
  if ( v22 )
    memset(v23, 0, 8LL * v22);
}

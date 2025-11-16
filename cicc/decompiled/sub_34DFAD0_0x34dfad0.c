// Function: sub_34DFAD0
// Address: 0x34dfad0
//
__int64 __fastcall sub_34DFAD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 (*v5)(void); // rdx
  __int64 v6; // rax
  char *v7; // r13
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // rdx
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
  int v21; // r14d
  __int64 result; // rax
  unsigned int v23; // r12d

  *(_QWORD *)a1 = off_49D8D38;
  v4 = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = v4;
  v5 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v6 = 0;
  if ( v5 != sub_2DAC790 )
  {
    v6 = v5();
    a2 = *(_QWORD *)(a1 + 8);
  }
  *(_QWORD *)(a1 + 24) = v6;
  v7 = 0;
  v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 32) = v8;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x600000000LL;
  *(_DWORD *)(a1 + 112) = 0;
  v10 = *(unsigned int *)(v8 + 16);
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  if ( v10 )
  {
    v11 = 8 * v10;
    v12 = (char *)sub_22077B0(8 * v10);
    v7 = &v12[v11];
    *(_QWORD *)(a1 + 120) = v12;
    *(_QWORD *)(a1 + 136) = &v12[v11];
    memset(v12, 0, v11);
    v8 = *(_QWORD *)(a1 + 32);
  }
  *(_QWORD *)(a1 + 128) = v7;
  v13 = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = a1 + 152;
  *(_QWORD *)(a1 + 176) = a1 + 152;
  *(_QWORD *)(a1 + 184) = 0;
  v14 = *(unsigned int *)(v8 + 16);
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  if ( v14 )
  {
    v15 = 4 * v14;
    v16 = (char *)sub_22077B0(4 * v14);
    v13 = &v16[v15];
    *(_QWORD *)(a1 + 192) = v16;
    *(_QWORD *)(a1 + 208) = &v16[v15];
    memset(v16, 0, v15);
    v8 = *(_QWORD *)(a1 + 32);
  }
  *(_QWORD *)(a1 + 200) = v13;
  v17 = *(unsigned int *)(v8 + 16);
  v18 = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  if ( v17 )
  {
    v19 = 4 * v17;
    v20 = (char *)sub_22077B0(4 * v17);
    v18 = &v20[v19];
    *(_QWORD *)(a1 + 216) = v20;
    *(_QWORD *)(a1 + 232) = &v20[v19];
    memset(v20, 0, v19);
    v8 = *(_QWORD *)(a1 + 32);
  }
  *(_QWORD *)(a1 + 224) = v18;
  v21 = *(_DWORD *)(v8 + 16);
  result = 0x600000000LL;
  *(_QWORD *)(a1 + 240) = a1 + 256;
  *(_QWORD *)(a1 + 248) = 0x600000000LL;
  v23 = (unsigned int)(v21 + 63) >> 6;
  if ( v23 > 6 )
  {
    sub_C8D5F0(a1 + 240, (const void *)(a1 + 256), v23, 8u, a1 + 240, v9);
    result = (__int64)memset(*(void **)(a1 + 240), 0, 8LL * v23);
    *(_DWORD *)(a1 + 248) = v23;
    *(_DWORD *)(a1 + 304) = v21;
  }
  else
  {
    if ( v23 )
      result = (__int64)memset((void *)(a1 + 256), 0, 8LL * v23);
    *(_DWORD *)(a1 + 248) = v23;
    *(_DWORD *)(a1 + 304) = v21;
  }
  return result;
}

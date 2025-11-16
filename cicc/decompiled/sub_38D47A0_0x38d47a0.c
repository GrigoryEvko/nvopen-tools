// Function: sub_38D47A0
// Address: 0x38d47a0
//
void __fastcall sub_38D47A0(__int64 a1, int *a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r14
  unsigned int v9; // r8d
  int v10; // edx
  void *v11; // rdi
  __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 *v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rax
  int v19; // r8d
  int v20; // r9d
  __int64 v21; // rdi
  size_t v22; // r13
  void *v23; // r15
  int v24; // eax
  int v25; // r12d
  size_t v26; // rdx
  unsigned int v27; // [rsp+8h] [rbp-F8h]
  unsigned int v28; // [rsp+8h] [rbp-F8h]
  _QWORD v29[4]; // [rsp+10h] [rbp-F0h] BYREF
  int v30; // [rsp+30h] [rbp-D0h]
  void **p_src; // [rsp+38h] [rbp-C8h]
  void *src; // [rsp+40h] [rbp-C0h] BYREF
  size_t n; // [rsp+48h] [rbp-B8h]
  _BYTE v34[176]; // [rsp+50h] [rbp-B0h] BYREF

  v6 = sub_22077B0(0x128u);
  v7 = v6;
  if ( v6 )
  {
    v8 = v6;
    sub_38CF760(v6, 4, 1, 0);
    v9 = a2[6];
    *(_QWORD *)(v7 + 88) = v7 + 104;
    v10 = *a2;
    v11 = (void *)(v7 + 160);
    *(_WORD *)(v7 + 48) = 0;
    *(_DWORD *)(v7 + 128) = v10;
    v12 = *((_QWORD *)a2 + 1);
    *(_QWORD *)(v7 + 64) = v7 + 80;
    *(_QWORD *)(v7 + 56) = 0;
    *(_QWORD *)(v7 + 72) = 0x800000000LL;
    *(_QWORD *)(v7 + 96) = 0x100000000LL;
    *(_QWORD *)(v7 + 136) = v12;
    *(_QWORD *)(v7 + 144) = v7 + 160;
    *(_QWORD *)(v7 + 152) = 0x800000000LL;
    if ( v9 && (int *)(v7 + 144) != a2 + 4 )
    {
      v26 = 16LL * v9;
      if ( v9 <= 8
        || (v28 = v9,
            sub_16CD150(v7 + 144, (const void *)(v7 + 160), v9, 16, v9, v7 + 144),
            v11 = *(void **)(v7 + 144),
            v9 = v28,
            (v26 = 16LL * (unsigned int)a2[6]) != 0) )
      {
        v27 = v9;
        memcpy(v11, *((const void **)a2 + 2), v26);
        v9 = v27;
      }
      *(_DWORD *)(v7 + 152) = v9;
    }
    v13 = a2[40];
    *(_QWORD *)(v7 + 56) = a3;
    *(_DWORD *)(v7 + 288) = v13;
  }
  else
  {
    v8 = 0;
  }
  sub_38D4150(a1, v7, 0);
  v14 = *(unsigned int *)(a1 + 120);
  v15 = 0;
  if ( (_DWORD)v14 )
    v15 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v14 - 32);
  v16 = *(__int64 **)(a1 + 272);
  v17 = *v16;
  v18 = *(_QWORD *)v7 & 7LL;
  *(_QWORD *)(v7 + 8) = v16;
  v17 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v7 = v17 | v18;
  *(_QWORD *)(v17 + 8) = v8;
  *v16 = *v16 & 7 | v8;
  *(_QWORD *)(v7 + 24) = v15;
  src = v34;
  n = 0x8000000000LL;
  v29[0] = &unk_49EFC48;
  p_src = &src;
  v30 = 1;
  memset(&v29[1], 0, 24);
  sub_16E7A40((__int64)v29, 0, 0, 0);
  (*(void (__fastcall **)(_QWORD, int *, _QWORD *, __int64, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 264) + 16LL) + 24LL))(
    *(_QWORD *)(*(_QWORD *)(a1 + 264) + 16LL),
    a2,
    v29,
    v7 + 88,
    a3);
  v21 = *(unsigned int *)(v7 + 72);
  v22 = (unsigned int)n;
  v23 = src;
  v24 = *(_DWORD *)(v7 + 72);
  v25 = n;
  if ( (unsigned int)n > (unsigned __int64)*(unsigned int *)(v7 + 76) - v21 )
  {
    sub_16CD150(v7 + 64, (const void *)(v7 + 80), (unsigned int)n + v21, 1, v19, v20);
    v21 = *(unsigned int *)(v7 + 72);
    v24 = *(_DWORD *)(v7 + 72);
  }
  if ( v25 )
  {
    memcpy((void *)(*(_QWORD *)(v7 + 64) + v21), v23, v22);
    v24 = *(_DWORD *)(v7 + 72);
  }
  *(_DWORD *)(v7 + 72) = v24 + v25;
  v29[0] = &unk_49EFD28;
  sub_16E7960((__int64)v29);
  if ( src != v34 )
    _libc_free((unsigned __int64)src);
}

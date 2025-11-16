// Function: sub_390A2E0
// Address: 0x390a2e0
//
__int64 __fastcall sub_390A2E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  int v5; // r9d
  unsigned __int64 v6; // rdx
  int v7; // r8d
  unsigned __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  int v11; // eax
  unsigned __int64 v12; // rdx
  size_t v13; // rcx
  int v14; // r8d
  unsigned int v15; // eax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  int v18; // r8d
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v22; // [rsp+8h] [rbp-2A8h]
  int v23; // [rsp+10h] [rbp-2A0h]
  int v24; // [rsp+10h] [rbp-2A0h]
  int v25; // [rsp+10h] [rbp-2A0h]
  int v26; // [rsp+18h] [rbp-298h]
  int v27; // [rsp+18h] [rbp-298h]
  int v28; // [rsp+18h] [rbp-298h]
  int v29; // [rsp+18h] [rbp-298h]
  int v30; // [rsp+18h] [rbp-298h]
  int v31; // [rsp+18h] [rbp-298h]
  int v32; // [rsp+18h] [rbp-298h]
  int v33; // [rsp+18h] [rbp-298h]
  int v34; // [rsp+18h] [rbp-298h]
  size_t v35; // [rsp+18h] [rbp-298h]
  __int64 v36; // [rsp+18h] [rbp-298h]
  __int64 v37; // [rsp+18h] [rbp-298h]
  _QWORD v38[4]; // [rsp+20h] [rbp-290h] BYREF
  int v39; // [rsp+40h] [rbp-270h]
  void **v40; // [rsp+48h] [rbp-268h]
  void *v41; // [rsp+50h] [rbp-260h] BYREF
  __int64 v42; // [rsp+58h] [rbp-258h]
  _BYTE v43[96]; // [rsp+60h] [rbp-250h] BYREF
  int v44; // [rsp+C0h] [rbp-1F0h] BYREF
  __int64 v45; // [rsp+C8h] [rbp-1E8h]
  void *src; // [rsp+D0h] [rbp-1E0h]
  __int64 v47; // [rsp+D8h] [rbp-1D8h]
  _BYTE v48[128]; // [rsp+E0h] [rbp-1D0h] BYREF
  int v49; // [rsp+160h] [rbp-150h]
  void *v50; // [rsp+170h] [rbp-140h] BYREF
  size_t n; // [rsp+178h] [rbp-138h]
  _BYTE v52[304]; // [rsp+180h] [rbp-130h] BYREF

  v3 = *(_QWORD *)(a1 + 8);
  v47 = 0x800000000LL;
  v44 = 0;
  v4 = *(_QWORD *)(a2 + 56);
  v45 = 0;
  src = v48;
  v49 = 0;
  v22 = a1;
  (*(void (__fastcall **)(__int64, __int64, __int64, int *))(*(_QWORD *)v3 + 104LL))(v3, a2 + 128, v4, &v44);
  v41 = v43;
  v42 = 0x400000000LL;
  v38[0] = &unk_49EFC48;
  v40 = &v50;
  n = 0x10000000000LL;
  v50 = v52;
  v39 = 1;
  memset(&v38[1], 0, 24);
  sub_16E7A40((__int64)v38, 0, 0, 0);
  (*(void (__fastcall **)(_QWORD, int *, _QWORD *, void **, _QWORD))(**(_QWORD **)(v22 + 16) + 24LL))(
    *(_QWORD *)(v22 + 16),
    &v44,
    v38,
    &v41,
    *(_QWORD *)(a2 + 56));
  v6 = (unsigned int)v47;
  *(_DWORD *)(a2 + 128) = v44;
  v7 = v6;
  *(_QWORD *)(a2 + 136) = v45;
  v8 = *(unsigned int *)(a2 + 152);
  if ( v6 <= v8 )
  {
    if ( v6 )
    {
      v29 = v6;
      memmove(*(void **)(a2 + 144), src, 16 * v6);
      v7 = v29;
    }
  }
  else
  {
    if ( v6 > *(unsigned int *)(a2 + 156) )
    {
      *(_DWORD *)(a2 + 152) = 0;
      v32 = v6;
      sub_16CD150(a2 + 144, (const void *)(a2 + 160), v6, 16, v6, v5);
      v6 = (unsigned int)v47;
      v7 = v32;
      v9 = 0;
    }
    else
    {
      v9 = 16 * v8;
      if ( *(_DWORD *)(a2 + 152) )
      {
        v24 = v6;
        v36 = 16 * v8;
        memmove(*(void **)(a2 + 144), src, 16 * v8);
        v6 = (unsigned int)v47;
        v7 = v24;
        v9 = v36;
      }
    }
    v10 = 16 * v6;
    if ( (char *)src + v9 != (char *)src + v10 )
    {
      v26 = v7;
      memcpy((void *)(v9 + *(_QWORD *)(a2 + 144)), (char *)src + v9, v10 - v9);
      v7 = v26;
    }
  }
  v11 = v49;
  v12 = (unsigned int)n;
  *(_DWORD *)(a2 + 152) = v7;
  v13 = *(unsigned int *)(a2 + 72);
  *(_DWORD *)(a2 + 288) = v11;
  v14 = v12;
  if ( v12 <= v13 )
  {
    if ( v12 )
    {
      v31 = v12;
      memmove(*(void **)(a2 + 64), v50, v12);
      v14 = v31;
    }
  }
  else
  {
    if ( v12 > *(unsigned int *)(a2 + 76) )
    {
      *(_DWORD *)(a2 + 72) = 0;
      v34 = v12;
      sub_16CD150(a2 + 64, (const void *)(a2 + 80), v12, 1, v12, v5);
      v12 = (unsigned int)n;
      v14 = v34;
      v13 = 0;
      v15 = n;
    }
    else
    {
      v15 = v12;
      if ( v13 )
      {
        v23 = v12;
        v35 = v13;
        memmove(*(void **)(a2 + 64), v50, v13);
        v12 = (unsigned int)n;
        v14 = v23;
        v13 = v35;
        v15 = n;
      }
    }
    if ( (char *)v50 + v13 != (char *)v50 + v12 )
    {
      v27 = v14;
      memcpy((void *)(v13 + *(_QWORD *)(a2 + 64)), (char *)v50 + v13, v15 - v13);
      v14 = v27;
    }
  }
  v16 = (unsigned int)v42;
  v17 = *(unsigned int *)(a2 + 96);
  *(_DWORD *)(a2 + 72) = v14;
  v18 = v16;
  if ( v16 <= v17 )
  {
    if ( v16 )
    {
      v30 = v16;
      memmove(*(void **)(a2 + 88), v41, 24 * v16);
      v18 = v30;
    }
  }
  else
  {
    if ( v16 > *(unsigned int *)(a2 + 100) )
    {
      *(_DWORD *)(a2 + 96) = 0;
      v33 = v16;
      sub_16CD150(a2 + 88, (const void *)(a2 + 104), v16, 24, v16, v5);
      v16 = (unsigned int)v42;
      v18 = v33;
      v19 = 0;
    }
    else
    {
      v19 = 24 * v17;
      if ( v17 )
      {
        v25 = v16;
        v37 = 24 * v17;
        memmove(*(void **)(a2 + 88), v41, 24 * v17);
        v16 = (unsigned int)v42;
        v18 = v25;
        v19 = v37;
      }
    }
    v20 = 24 * v16;
    if ( (char *)v41 + v19 != (char *)v41 + v20 )
    {
      v28 = v18;
      memcpy((void *)(v19 + *(_QWORD *)(a2 + 88)), (char *)v41 + v19, v20 - v19);
      v18 = v28;
    }
  }
  *(_DWORD *)(a2 + 96) = v18;
  v38[0] = &unk_49EFD28;
  sub_16E7960((__int64)v38);
  if ( v50 != v52 )
    _libc_free((unsigned __int64)v50);
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
  if ( src != v48 )
    _libc_free((unsigned __int64)src);
  return 1;
}

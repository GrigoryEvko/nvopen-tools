// Function: sub_20C2AF0
// Address: 0x20c2af0
//
void __fastcall sub_20C2AF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 (*v8)(void); // rdx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 (*v12)(); // rax
  int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // rcx
  int v16; // r8d
  int v17; // r9d
  unsigned int v18; // edx
  _QWORD *v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // rdx
  unsigned int v23; // r9d
  unsigned int v24; // r9d
  __int64 v25; // r10
  unsigned __int64 v26; // rdx
  int v27; // ecx
  int v28; // ecx
  __int64 v29; // rax
  size_t v30; // rdx
  __int64 v31; // r10
  void *v32; // rcx
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  void *v35; // [rsp+0h] [rbp-70h]
  __int64 v36; // [rsp+8h] [rbp-68h]
  __int64 v37; // [rsp+8h] [rbp-68h]
  __int64 v38; // [rsp+8h] [rbp-68h]
  size_t n; // [rsp+10h] [rbp-60h]
  size_t na; // [rsp+10h] [rbp-60h]
  unsigned int nb; // [rsp+10h] [rbp-60h]
  size_t nc; // [rsp+10h] [rbp-60h]
  __int64 v43; // [rsp+18h] [rbp-58h]
  void *src[2]; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v45; // [rsp+30h] [rbp-40h]

  *(_QWORD *)a1 = off_49859B8;
  v7 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = v7;
  v8 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v9 = 0;
  if ( v8 != sub_1D00B00 )
  {
    v9 = v8();
    a2 = *(_QWORD *)(a1 + 8);
  }
  *(_QWORD *)(a1 + 24) = v9;
  v10 = *(_QWORD *)(a2 + 16);
  v11 = 0;
  v12 = *(__int64 (**)())(*(_QWORD *)v10 + 112LL);
  if ( v12 != sub_1D00B10 )
    v11 = ((__int64 (__fastcall *)(__int64, _QWORD))v12)(v10, 0);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  v13 = *(_DWORD *)(a4 + 8);
  *(_QWORD *)(a1 + 32) = v11;
  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 72) = 0;
  if ( v13 )
  {
    v14 = 0;
    v43 = 8LL * (unsigned int)(v13 - 1);
    while ( 1 )
    {
      sub_1F4AD00((__int64)src, v11, *(_QWORD *)(a1 + 8), *(_QWORD **)(*(_QWORD *)a4 + v14));
      v18 = (unsigned int)(*(_DWORD *)(a1 + 64) + 63) >> 6;
      if ( v18 )
      {
        v19 = *(_QWORD **)(a1 + 48);
        v20 = (__int64)&v19[v18];
        while ( !*v19 )
        {
          if ( ++v19 == (_QWORD *)v20 )
            goto LABEL_14;
        }
        sub_20C1FD0(a1 + 48, (__int64)src, v20, v15, v16, v17);
LABEL_12:
        _libc_free((unsigned __int64)src[0]);
        if ( v43 == v14 )
          return;
        goto LABEL_13;
      }
LABEL_14:
      if ( (void **)(a1 + 48) == src )
        goto LABEL_12;
      v21 = v45;
      v22 = *(_QWORD *)(a1 + 56);
      v23 = v45 + 63;
      *(_DWORD *)(a1 + 64) = v45;
      v24 = v23 >> 6;
      v25 = v24;
      if ( v21 > v22 << 6 )
      {
        v36 = v24;
        n = 8LL * v24;
        v29 = malloc(n);
        v30 = n;
        v31 = v36;
        v32 = (void *)v29;
        if ( !v29 )
        {
          if ( n || (v34 = malloc(1u), v30 = 0, v31 = v36, v32 = 0, !v34) )
          {
            v35 = v32;
            v38 = v31;
            nc = v30;
            sub_16BD1C0("Allocation failed", 1u);
            v30 = nc;
            v31 = v38;
            v32 = v35;
          }
          else
          {
            v32 = (void *)v34;
          }
        }
        v37 = v31;
        na = (size_t)v32;
        memcpy(v32, src[0], v30);
        _libc_free(*(_QWORD *)(a1 + 48));
        *(_QWORD *)(a1 + 48) = na;
        *(_QWORD *)(a1 + 56) = v37;
        goto LABEL_12;
      }
      if ( !(_DWORD)v21 )
        break;
      memcpy(*(void **)(a1 + 48), src[0], 8LL * v24);
      v27 = *(_DWORD *)(a1 + 64);
      v33 = *(_QWORD *)(a1 + 56);
      v24 = (unsigned int)(v27 + 63) >> 6;
      v25 = v24;
      if ( v33 > v24 )
      {
        v26 = v33 - v24;
        if ( v26 )
          goto LABEL_27;
        goto LABEL_19;
      }
LABEL_20:
      v28 = v27 & 0x3F;
      if ( !v28 )
        goto LABEL_12;
      *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * (v24 - 1)) &= ~(-1LL << v28);
      _libc_free((unsigned __int64)src[0]);
      if ( v43 == v14 )
        return;
LABEL_13:
      v11 = *(_QWORD *)(a1 + 32);
      v14 += 8;
    }
    if ( v22 <= v24 )
      goto LABEL_12;
    v26 = v22 - v24;
    if ( v26 )
    {
LABEL_27:
      nb = v24;
      memset((void *)(*(_QWORD *)(a1 + 48) + 8 * v25), 0, 8 * v26);
      v24 = nb;
    }
LABEL_19:
    v27 = *(_DWORD *)(a1 + 64);
    goto LABEL_20;
  }
}

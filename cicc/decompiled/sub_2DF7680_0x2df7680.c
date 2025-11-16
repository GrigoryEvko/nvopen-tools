// Function: sub_2DF7680
// Address: 0x2df7680
//
void __fastcall sub_2DF7680(__int64 a1, __int64 a2)
{
  void *v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rbx
  char v6; // dl
  char v7; // al
  char v8; // al
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  unsigned int v12; // eax
  __int64 v13; // rdx
  bool v14; // bl
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // rbx
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rsi
  int v23; // eax
  unsigned int v24; // r8d
  __int64 v25; // rbx
  __int64 v26; // rax
  void *v27; // rdi
  void *v28; // r8
  size_t v29; // rax
  __int64 v30; // [rsp+0h] [rbp-60h]
  __int64 v31; // [rsp+0h] [rbp-60h]
  __int64 v32; // [rsp+8h] [rbp-58h]
  _QWORD *v33; // [rsp+8h] [rbp-58h]
  unsigned int v34; // [rsp+8h] [rbp-58h]
  void *src; // [rsp+10h] [rbp-50h] BYREF
  char v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+20h] [rbp-40h]

  sub_2DF52D0((__int64)&src, a2);
  v3 = src;
  v4 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v5 = *(_QWORD *)v4 + 24LL * *(unsigned int *)(v4 + 12) + 64;
  if ( (void **)v5 != &src )
  {
    if ( (v36 & 0x3F) != 0 )
    {
      v26 = sub_2207820(4LL * (v36 & 0x3F));
      v27 = *(void **)v5;
      v28 = (void *)v26;
      *(_QWORD *)v5 = v26;
      if ( v27 )
      {
        j_j___libc_free_0_0((unsigned __int64)v27);
        v28 = *(void **)v5;
      }
      v3 = src;
      v6 = v36 & 0x3F;
      v29 = 4LL * (v36 & 0x3F);
      if ( v29 )
      {
        memmove(v28, src, v29);
        v3 = src;
        v6 = v36 & 0x3F;
      }
    }
    else
    {
      *(_QWORD *)v5 = 0;
      v3 = src;
      v6 = v36 & 0x3F;
    }
    v7 = v6 | *(_BYTE *)(v5 + 8) & 0xC0;
    *(_BYTE *)(v5 + 8) = v7;
    v8 = v36 & 0x40 | v7 & 0xBF;
    *(_BYTE *)(v5 + 8) = v8;
    *(_BYTE *)(v5 + 8) = v36 & 0x80 | v8 & 0x7F;
    *(_QWORD *)(v5 + 16) = v37;
  }
  if ( v3 )
    j_j___libc_free_0_0((unsigned __int64)v3);
  sub_2DF52D0((__int64)&src, a2);
  v9 = *(unsigned int *)(a1 + 16);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 160LL) )
  {
    v10 = *(_QWORD *)(a1 + 8) + 16 * v9 - 16;
    v11 = *(_QWORD *)(*(_QWORD *)v10 + 16LL * *(unsigned int *)(v10 + 12) + 8);
    v12 = *(_DWORD *)(v10 + 12) + 1;
    if ( *(_DWORD *)(v10 + 8) <= v12 )
    {
      v15 = sub_F03C90((__int64 *)(a1 + 8), *(_DWORD *)(a1 + 16) - 1);
      if ( v15 )
      {
        v33 = (_QWORD *)(v15 & 0xFFFFFFFFFFFFFFC0LL);
        if ( sub_2DF4840((v15 & 0xFFFFFFFFFFFFFFC0LL) + 64, (__int64)&src) && *v33 == v11 )
          goto LABEL_19;
      }
    }
    else
    {
      v32 = *(_QWORD *)v10;
      v30 = v12;
      if ( sub_2DF4840(*(_QWORD *)v10 + 24LL * v12 + 64, (__int64)&src) && *(_QWORD *)(v32 + 16 * v30) == v11 )
      {
LABEL_19:
        if ( src )
          j_j___libc_free_0_0((unsigned __int64)src);
        v16 = *(_QWORD *)sub_2DF4990(a1);
        sub_2DF6D10(a1);
        *(_QWORD *)sub_2DF4990(a1) = v16;
        goto LABEL_12;
      }
    }
  }
  else
  {
    v17 = *(_QWORD *)(a1 + 8) + 16 * v9 - 16;
    v34 = *(_DWORD *)(v17 + 12);
    v18 = v34 + 1;
    if ( *(_DWORD *)(v17 + 8) > v34 + 1 )
    {
      v19 = v18;
      v31 = *(_QWORD *)v17;
      if ( sub_2DF4840(*(_QWORD *)v17 + 24LL * v18 + 64, (__int64)&src)
        && *(_QWORD *)(v31 + 16 * v19) == *(_QWORD *)(v31 + 16LL * v34 + 8) )
      {
        goto LABEL_19;
      }
    }
  }
  if ( src )
    j_j___libc_free_0_0((unsigned __int64)src);
LABEL_12:
  sub_2DF52D0((__int64)&src, a2);
  v13 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v14 = sub_2DF48C0(a1, *(_QWORD *)(*(_QWORD *)v13 + 16LL * *(unsigned int *)(v13 + 12)), (__int64)&src);
  if ( src )
    j_j___libc_free_0_0((unsigned __int64)src);
  if ( v14 )
  {
    v20 = *(_QWORD *)(a1 + 8);
    v21 = *(unsigned int *)(a1 + 16);
    v22 = v20 + 16 * v21 - 16;
    v23 = *(_DWORD *)(v22 + 12);
    if ( v23 )
    {
      if ( (_DWORD)v21 && *(_DWORD *)(v20 + 12) < *(_DWORD *)(v20 + 8)
        || (v24 = *(_DWORD *)(*(_QWORD *)a1 + 160LL)) == 0 )
      {
        *(_DWORD *)(v22 + 12) = v23 - 1;
LABEL_30:
        v25 = *(_QWORD *)sub_2DF4990(a1);
        sub_2DF6D10(a1);
        *(_QWORD *)sub_2DF4990(a1) = v25;
        return;
      }
    }
    else
    {
      v24 = *(_DWORD *)(*(_QWORD *)a1 + 160LL);
    }
    sub_F03AD0((unsigned int *)(a1 + 8), v24);
    goto LABEL_30;
  }
}

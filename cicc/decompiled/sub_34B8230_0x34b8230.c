// Function: sub_34B8230
// Address: 0x34b8230
//
__int64 __fastcall sub_34B8230(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r9
  __int64 v8; // r14
  unsigned __int16 ***v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // r15
  __int64 v12; // r10
  char v13; // al
  __int64 v14; // r9
  unsigned __int64 v15; // r14
  void *v16; // r8
  char v17; // r13
  __int64 v18; // r10
  int v19; // r11d
  _BYTE *v20; // rdi
  int v21; // eax
  unsigned __int64 v23; // rdx
  size_t v24; // rdx
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-88h]
  __int64 v27; // [rsp+10h] [rbp-80h]
  __int64 v28; // [rsp+10h] [rbp-80h]
  __int64 v29; // [rsp+18h] [rbp-78h]
  __int64 v30; // [rsp+18h] [rbp-78h]
  int v31; // [rsp+18h] [rbp-78h]
  _BYTE *v32; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v33; // [rsp+28h] [rbp-68h]
  __int64 v34; // [rsp+30h] [rbp-60h]
  _BYTE src[88]; // [rsp+38h] [rbp-58h] BYREF

  v7 = *(_QWORD *)(a3 + 24);
  v8 = *(_QWORD *)(a3 + 16);
  v9 = (unsigned __int16 ***)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 32) + 56LL) + 16LL * (a2 & 0x7FFFFFFF))
                            & 0xFFFFFFFFFFFFFFF8LL);
  v10 = *(_QWORD *)a4 + 24LL * *((unsigned __int16 *)*v9 + 12);
  if ( *(_DWORD *)(a4 + 8) != *(_DWORD *)v10 )
  {
    v26 = a5;
    v27 = a3;
    v29 = *(_QWORD *)(a3 + 24);
    sub_2F60630(a4, v9);
    a5 = v26;
    a3 = v27;
    v7 = v29;
  }
  v11 = *(unsigned int *)(v10 + 4);
  v12 = *(_QWORD *)(v10 + 16);
  v32 = src;
  v33 = 0;
  v34 = 16;
  v30 = v12;
  v13 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64, _BYTE **, __int64, __int64, __int64))(*(_QWORD *)v8 + 448LL))(
          v8,
          a2,
          v12,
          v11,
          &v32,
          v7,
          a3,
          a5);
  v15 = v33;
  v16 = (void *)(a1 + 24);
  *(_QWORD *)(a1 + 8) = 0;
  v17 = v13;
  v18 = v30;
  *(_QWORD *)a1 = a1 + 24;
  v19 = v11;
  *(_QWORD *)(a1 + 16) = 16;
  if ( !v15 )
  {
    v20 = v32;
    goto LABEL_5;
  }
  if ( v32 == src )
  {
    v23 = v15;
    v20 = src;
    if ( v15 > 0x10 )
    {
      sub_C8D290(a1, (const void *)(a1 + 24), v15, 2u, (__int64)v16, v14);
      v16 = *(void **)a1;
      v20 = v32;
      v23 = v33;
      v18 = v30;
      v19 = v11;
    }
    v24 = 2 * v23;
    if ( v24 )
    {
      v28 = v18;
      v31 = v19;
      memcpy(v16, v20, v24);
      v20 = v32;
      v18 = v28;
      v19 = v31;
    }
    *(_QWORD *)(a1 + 8) = v15;
LABEL_5:
    *(_QWORD *)(a1 + 56) = v18;
    v21 = 0;
    *(_QWORD *)(a1 + 64) = v11;
    if ( v17 )
      goto LABEL_6;
    goto LABEL_9;
  }
  *(_QWORD *)a1 = v32;
  v25 = v34;
  v20 = src;
  *(_QWORD *)(a1 + 8) = v15;
  *(_QWORD *)(a1 + 16) = v25;
  *(_QWORD *)(a1 + 56) = v30;
  *(_QWORD *)(a1 + 64) = v11;
  if ( v17 )
  {
    *(_DWORD *)(a1 + 72) = 0;
    return a1;
  }
LABEL_9:
  v21 = v19;
LABEL_6:
  *(_DWORD *)(a1 + 72) = v21;
  if ( v20 != src )
    _libc_free((unsigned __int64)v20);
  return a1;
}

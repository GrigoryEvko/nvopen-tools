// Function: sub_2713180
// Address: 0x2713180
//
void __fastcall sub_2713180(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // eax
  __int64 v4; // rbx
  __int64 v5; // rcx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // rcx
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r8
  __int64 *v18; // r13
  __int64 *v19; // r14
  unsigned int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // r10
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rdi
  int v26; // eax
  int v27; // r11d
  __int64 v28; // r14
  __int64 v29; // r13
  __int64 v30; // rbx
  __int64 v31; // r13
  __int64 v32; // rbx
  __int64 v33; // [rsp-F8h] [rbp-F8h]
  __int64 v34; // [rsp-E8h] [rbp-E8h]
  __int64 v35; // [rsp-D0h] [rbp-D0h]
  __int64 v36; // [rsp-D0h] [rbp-D0h]
  __int64 v37; // [rsp-D0h] [rbp-D0h]
  __int64 v38; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v39; // [rsp-C0h] [rbp-C0h] BYREF
  __int64 v40; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v41; // [rsp-B0h] [rbp-B0h]
  __int64 v42; // [rsp-A8h] [rbp-A8h]
  __int64 v43; // [rsp-A0h] [rbp-A0h] BYREF
  _BYTE *v44; // [rsp-98h] [rbp-98h]
  __int64 v45; // [rsp-90h] [rbp-90h]
  int v46; // [rsp-88h] [rbp-88h]
  char v47; // [rsp-84h] [rbp-84h]
  _BYTE v48[16]; // [rsp-80h] [rbp-80h] BYREF
  __int64 v49; // [rsp-70h] [rbp-70h] BYREF
  _BYTE *v50; // [rsp-68h] [rbp-68h]
  __int64 v51; // [rsp-60h] [rbp-60h]
  int v52; // [rsp-58h] [rbp-58h]
  char v53; // [rsp-54h] [rbp-54h]
  _BYTE v54[16]; // [rsp-50h] [rbp-50h] BYREF
  char v55; // [rsp-40h] [rbp-40h]

  if ( *(_DWORD *)a1 == -1 )
    return;
  v2 = a2;
  v3 = *(_DWORD *)a2 + *(_DWORD *)a1;
  *(_DWORD *)a1 = v3;
  if ( v3 == -1 )
  {
    sub_270F800(a1 + 8);
    v28 = *(_QWORD *)(a1 + 40);
    v31 = *(_QWORD *)(a1 + 48);
    if ( v28 == v31 )
      return;
    v32 = *(_QWORD *)(a1 + 40);
    while ( 1 )
    {
      if ( *(_BYTE *)(v32 + 108) )
      {
        if ( !*(_BYTE *)(v32 + 60) )
          goto LABEL_50;
      }
      else
      {
        _libc_free(*(_QWORD *)(v32 + 88));
        if ( !*(_BYTE *)(v32 + 60) )
LABEL_50:
          _libc_free(*(_QWORD *)(v32 + 40));
      }
      v32 += 136;
      if ( v31 == v32 )
      {
LABEL_43:
        *(_QWORD *)(a1 + 48) = v28;
        return;
      }
    }
  }
  if ( v3 < *(_DWORD *)a2 )
  {
    *(_DWORD *)a1 = -1;
    sub_270F800(a1 + 8);
    v28 = *(_QWORD *)(a1 + 40);
    v29 = *(_QWORD *)(a1 + 48);
    if ( v28 == v29 )
      return;
    v30 = *(_QWORD *)(a1 + 40);
    while ( 1 )
    {
      if ( *(_BYTE *)(v30 + 108) )
      {
        if ( !*(_BYTE *)(v30 + 60) )
          goto LABEL_42;
      }
      else
      {
        _libc_free(*(_QWORD *)(v30 + 88));
        if ( !*(_BYTE *)(v30 + 60) )
LABEL_42:
          _libc_free(*(_QWORD *)(v30 + 40));
      }
      v30 += 136;
      if ( v29 == v30 )
        goto LABEL_43;
    }
  }
  v34 = *(_QWORD *)(a2 + 48);
  if ( *(_QWORD *)(a2 + 40) != v34 )
  {
    v4 = *(_QWORD *)(a2 + 40);
    do
    {
      v10 = *(_QWORD *)v4;
      v39 = 0;
      v38 = v10;
      sub_2712ED0((__int64)&v40, a1 + 8, &v38, &v39);
      if ( (_BYTE)v44 )
      {
        v12 = *(_QWORD *)(a1 + 48) - *(_QWORD *)(a1 + 40);
        v13 = 0xF0F0F0F0F0F0F0F1LL * (v12 >> 3);
        *(_QWORD *)(v42 + 8) = v13;
        v14 = *(_QWORD *)(a1 + 48);
        if ( v14 == *(_QWORD *)(a1 + 56) )
        {
          v37 = v12;
          sub_270F9D0((__int64 *)(a1 + 40), *(_QWORD *)(a1 + 48), v4, v13, v12, v14);
          v12 = v37;
        }
        else
        {
          if ( v14 )
          {
            v33 = v12;
            v36 = *(_QWORD *)(a1 + 48);
            *(_QWORD *)v14 = *(_QWORD *)v4;
            *(_BYTE *)(v14 + 8) = *(_BYTE *)(v4 + 8);
            *(_BYTE *)(v14 + 9) = *(_BYTE *)(v4 + 9);
            *(_BYTE *)(v14 + 10) = *(_BYTE *)(v4 + 10);
            *(_BYTE *)(v14 + 16) = *(_BYTE *)(v4 + 16);
            *(_BYTE *)(v14 + 17) = *(_BYTE *)(v4 + 17);
            *(_QWORD *)(v14 + 24) = *(_QWORD *)(v4 + 24);
            sub_C8CD80(v14 + 32, v14 + 64, v4 + 32, v13, v12, v14);
            sub_C8CD80(v36 + 80, v36 + 112, v4 + 80, v15, v16, v36);
            v12 = v33;
            *(_BYTE *)(v36 + 128) = *(_BYTE *)(v4 + 128);
            v14 = *(_QWORD *)(a1 + 48);
          }
          *(_QWORD *)(a1 + 48) = v14 + 136;
        }
        v17 = *(_QWORD *)(a1 + 40) + v12;
        v40 = 0;
        v41 = 0;
        v9 = v17 + 8;
        v44 = v48;
        v42 = 0;
        v43 = 0;
        v45 = 2;
        v46 = 0;
        v47 = 1;
        v49 = 0;
        v50 = v54;
        v51 = 2;
        v52 = 0;
        v53 = 1;
        v55 = 0;
      }
      else
      {
        v5 = *(_QWORD *)(a1 + 40);
        v35 = v5 + 136LL * *(_QWORD *)(v42 + 8) + 8;
        LOWORD(v40) = *(_WORD *)(v4 + 8);
        BYTE2(v40) = *(_BYTE *)(v4 + 10);
        LOWORD(v41) = *(_WORD *)(v4 + 16);
        v42 = *(_QWORD *)(v4 + 24);
        sub_C8CD80((__int64)&v43, (__int64)v48, v4 + 32, v5, v35, v11);
        sub_C8CD80((__int64)&v49, (__int64)v54, v4 + 80, v6, v7, v8);
        v9 = v35;
        v55 = *(_BYTE *)(v4 + 128);
      }
      sub_271D550(v9, &v40, 1);
      if ( !v53 )
        _libc_free((unsigned __int64)v50);
      if ( !v47 )
        _libc_free((unsigned __int64)v44);
      v4 += 136;
    }
    while ( v34 != v4 );
    v2 = a2;
  }
  v18 = *(__int64 **)(a1 + 48);
  v19 = *(__int64 **)(a1 + 40);
  if ( v18 != v19 )
  {
    while ( 1 )
    {
      v23 = *(unsigned int *)(v2 + 32);
      v24 = *v19;
      v25 = *(_QWORD *)(v2 + 16);
      if ( !(_DWORD)v23 )
        goto LABEL_26;
      v20 = (v23 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v21 = (__int64 *)(v25 + 16LL * v20);
      v22 = *v21;
      if ( v24 != *v21 )
        break;
LABEL_22:
      if ( v21 == (__int64 *)(v25 + 16 * v23) || *(_QWORD *)(v2 + 48) == *(_QWORD *)(v2 + 40) + 136 * v21[1] )
        goto LABEL_26;
LABEL_24:
      v19 += 17;
      if ( v18 == v19 )
        return;
    }
    v26 = 1;
    while ( v22 != -4096 )
    {
      v27 = v26 + 1;
      v20 = (v23 - 1) & (v26 + v20);
      v21 = (__int64 *)(v25 + 16LL * v20);
      v22 = *v21;
      if ( v24 == *v21 )
        goto LABEL_22;
      v26 = v27;
    }
LABEL_26:
    v44 = v48;
    v40 = 0;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    v45 = 2;
    v46 = 0;
    v47 = 1;
    v49 = 0;
    v50 = v54;
    v51 = 2;
    v52 = 0;
    v53 = 1;
    v55 = 0;
    sub_271D550(v19 + 1, &v40, 1);
    if ( !v53 )
      _libc_free((unsigned __int64)v50);
    if ( !v47 )
      _libc_free((unsigned __int64)v44);
    goto LABEL_24;
  }
}

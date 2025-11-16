// Function: sub_A043F0
// Address: 0xa043f0
//
__int64 __fastcall sub_A043F0(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // r8
  __int64 v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rbx
  volatile signed __int32 *v11; // r12
  signed __int32 v12; // eax
  void (*v13)(); // rax
  signed __int32 v14; // eax
  __int64 (__fastcall *v15)(__int64); // rcx
  __int64 v16; // rbx
  __int64 v17; // r12
  volatile signed __int32 *v18; // r14
  signed __int32 v19; // eax
  void (*v20)(); // rax
  signed __int32 v21; // eax
  __int64 (__fastcall *v22)(__int64); // rdx
  void (__fastcall *v23)(__int64, __int64, __int64); // rax
  void (__fastcall *v24)(__int64, __int64, __int64); // rax
  __int64 *v25; // rbx
  __int64 *v26; // r12
  __int64 v27; // rdi
  _QWORD *v28; // r12
  _QWORD *v29; // rbx
  __int64 v30; // rbx
  __int64 result; // rax
  __int64 v32; // r12
  __int64 v33; // rax
  void (__fastcall *v34)(__int64, __int64, __int64); // rax
  __int64 v35; // [rsp+0h] [rbp-40h]
  __int64 v36; // [rsp+0h] [rbp-40h]
  __int64 v37; // [rsp+8h] [rbp-38h]

  sub_C7D6A0(*(_QWORD *)(a1 + 1112), 16LL * *(unsigned int *)(a1 + 1128), 8);
  v2 = 8LL * *(unsigned int *)(a1 + 1088);
  sub_C7D6A0(*(_QWORD *)(a1 + 1072), v2, 4);
  if ( (*(_BYTE *)(a1 + 800) & 1) == 0 )
  {
    v2 = 16LL * *(unsigned int *)(a1 + 816);
    sub_C7D6A0(*(_QWORD *)(a1 + 808), v2, 8);
  }
  v3 = *(_QWORD *)(a1 + 768);
  if ( v3 )
  {
    v2 = *(_QWORD *)(a1 + 784) - v3;
    j_j___libc_free_0(v3, v2);
  }
  v4 = *(_QWORD *)(a1 + 736);
  if ( v4 )
  {
    v2 = *(_QWORD *)(a1 + 752) - v4;
    j_j___libc_free_0(v4, v2);
  }
  v5 = *(_QWORD *)(a1 + 712);
  if ( v5 )
  {
    v2 = *(_QWORD *)(a1 + 728) - v5;
    j_j___libc_free_0(v5, v2);
  }
  v6 = 32LL * *(unsigned int *)(a1 + 440);
  v37 = *(_QWORD *)(a1 + 432);
  v7 = v37 + v6;
  if ( v37 != v37 + v6 )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v7 - 24);
      v9 = *(_QWORD *)(v7 - 16);
      v7 -= 32;
      v10 = v8;
      if ( v9 != v8 )
        break;
LABEL_25:
      if ( v8 )
      {
        v2 = *(_QWORD *)(v7 + 24) - v8;
        j_j___libc_free_0(v8, v2);
      }
      if ( v37 == v7 )
      {
        v7 = *(_QWORD *)(a1 + 432);
        goto LABEL_29;
      }
    }
    while ( 1 )
    {
      v11 = *(volatile signed __int32 **)(v10 + 8);
      if ( !v11 )
        goto LABEL_12;
      if ( &_pthread_key_create )
      {
        v12 = _InterlockedExchangeAdd(v11 + 2, 0xFFFFFFFF);
      }
      else
      {
        v12 = *((_DWORD *)v11 + 2);
        *((_DWORD *)v11 + 2) = v12 - 1;
      }
      if ( v12 != 1 )
        goto LABEL_12;
      v13 = *(void (**)())(*(_QWORD *)v11 + 16LL);
      if ( v13 != nullsub_25 )
      {
        v36 = v9;
        ((void (__fastcall *)(volatile signed __int32 *))v13)(v11);
        v9 = v36;
      }
      if ( &_pthread_key_create )
      {
        v14 = _InterlockedExchangeAdd(v11 + 3, 0xFFFFFFFF);
      }
      else
      {
        v14 = *((_DWORD *)v11 + 3);
        *((_DWORD *)v11 + 3) = v14 - 1;
      }
      if ( v14 != 1 )
        goto LABEL_12;
      v35 = v9;
      v15 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 24LL);
      if ( v15 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v11 + 8LL))(v11);
        v9 = v35;
        v10 += 16;
        if ( v35 == v10 )
        {
LABEL_24:
          v8 = *(_QWORD *)(v7 + 8);
          goto LABEL_25;
        }
      }
      else
      {
        v15((__int64)v11);
        v9 = v35;
LABEL_12:
        v10 += 16;
        if ( v9 == v10 )
          goto LABEL_24;
      }
    }
  }
LABEL_29:
  if ( v7 != a1 + 448 )
    _libc_free(v7, v2);
  v16 = *(_QWORD *)(a1 + 416);
  v17 = *(_QWORD *)(a1 + 408);
  if ( v16 != v17 )
  {
    while ( 1 )
    {
      v18 = *(volatile signed __int32 **)(v17 + 8);
      if ( !v18 )
        goto LABEL_33;
      if ( &_pthread_key_create )
      {
        v19 = _InterlockedExchangeAdd(v18 + 2, 0xFFFFFFFF);
      }
      else
      {
        v19 = *((_DWORD *)v18 + 2);
        *((_DWORD *)v18 + 2) = v19 - 1;
      }
      if ( v19 != 1 )
        goto LABEL_33;
      v20 = *(void (**)())(*(_QWORD *)v18 + 16LL);
      if ( v20 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v20)(v18);
      if ( &_pthread_key_create )
      {
        v21 = _InterlockedExchangeAdd(v18 + 3, 0xFFFFFFFF);
      }
      else
      {
        v21 = *((_DWORD *)v18 + 3);
        *((_DWORD *)v18 + 3) = v21 - 1;
      }
      if ( v21 != 1 )
        goto LABEL_33;
      v22 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v18 + 24LL);
      if ( v22 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v18 + 8LL))(v18);
        v17 += 16;
        if ( v16 == v17 )
        {
LABEL_45:
          v17 = *(_QWORD *)(a1 + 408);
          break;
        }
      }
      else
      {
        v22((__int64)v18);
LABEL_33:
        v17 += 16;
        if ( v16 == v17 )
          goto LABEL_45;
      }
    }
  }
  if ( v17 )
  {
    v2 = *(_QWORD *)(a1 + 424) - v17;
    j_j___libc_free_0(v17, v2);
  }
  if ( *(_BYTE *)(a1 + 360) )
  {
    v34 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 344);
    *(_BYTE *)(a1 + 360) = 0;
    if ( v34 )
    {
      v2 = a1 + 328;
      v34(a1 + 328, a1 + 328, 3);
    }
  }
  v23 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 312);
  if ( v23 )
  {
    v2 = a1 + 296;
    v23(a1 + 296, a1 + 296, 3);
  }
  v24 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 280);
  if ( v24 )
  {
    v2 = a1 + 264;
    v24(a1 + 264, a1 + 264, 3);
  }
  v25 = *(__int64 **)(a1 + 184);
  v26 = &v25[2 * *(unsigned int *)(a1 + 192)];
  if ( v25 != v26 )
  {
    do
    {
      v27 = *(v26 - 1);
      v26 -= 2;
      if ( v27 )
        sub_BA65D0();
      v2 = *v26;
      if ( *v26 )
        sub_B91220(v26);
    }
    while ( v25 != v26 );
    v26 = *(__int64 **)(a1 + 184);
  }
  if ( v26 != (__int64 *)(a1 + 200) )
    _libc_free(v26, v2);
  if ( (*(_BYTE *)(a1 + 160) & 1) == 0 )
  {
    v2 = 16LL * *(unsigned int *)(a1 + 176);
    sub_C7D6A0(*(_QWORD *)(a1 + 168), v2, 8);
  }
  if ( (*(_BYTE *)(a1 + 128) & 1) == 0 )
  {
    v2 = 16LL * *(unsigned int *)(a1 + 144);
    sub_C7D6A0(*(_QWORD *)(a1 + 136), v2, 8);
  }
  if ( (*(_BYTE *)(a1 + 96) & 1) != 0 )
  {
    v28 = (_QWORD *)(a1 + 104);
    v29 = (_QWORD *)(a1 + 120);
  }
  else
  {
    v28 = *(_QWORD **)(a1 + 104);
    v33 = *(unsigned int *)(a1 + 112);
    v2 = 16 * v33;
    if ( !(_DWORD)v33 )
      goto LABEL_93;
    v29 = (_QWORD *)((char *)v28 + v2);
    if ( (_QWORD *)((char *)v28 + v2) == v28 )
      goto LABEL_93;
  }
  do
  {
    if ( *v28 != -8192 && *v28 != -4096 && v28[1] )
      sub_BA65D0();
    v28 += 2;
  }
  while ( v29 != v28 );
  if ( (*(_BYTE *)(a1 + 96) & 1) == 0 )
  {
    v28 = *(_QWORD **)(a1 + 104);
    v2 = 16LL * *(unsigned int *)(a1 + 112);
LABEL_93:
    sub_C7D6A0(v28, v2, 8);
  }
  if ( (*(_BYTE *)(a1 + 64) & 1) == 0 )
  {
    v2 = 4LL * *(unsigned int *)(a1 + 80);
    sub_C7D6A0(*(_QWORD *)(a1 + 72), v2, 4);
  }
  if ( (*(_BYTE *)(a1 + 32) & 1) == 0 )
  {
    v2 = 4LL * *(unsigned int *)(a1 + 48);
    sub_C7D6A0(*(_QWORD *)(a1 + 40), v2, 4);
  }
  v30 = *(_QWORD *)a1;
  result = *(unsigned int *)(a1 + 8);
  v32 = *(_QWORD *)a1 + 8 * result;
  if ( *(_QWORD *)a1 != v32 )
  {
    do
    {
      v2 = *(_QWORD *)(v32 - 8);
      v32 -= 8;
      if ( v2 )
        result = sub_B91220(v32);
    }
    while ( v30 != v32 );
    v32 = *(_QWORD *)a1;
  }
  if ( v32 != a1 + 16 )
    return _libc_free(v32, v2);
  return result;
}

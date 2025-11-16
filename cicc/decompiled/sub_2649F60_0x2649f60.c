// Function: sub_2649F60
// Address: 0x2649f60
//
__int64 __fastcall sub_2649F60(__int64 a1, __int64 *a2)
{
  _QWORD *v3; // rax
  unsigned __int64 *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // r8
  unsigned __int64 v8; // r9
  __int64 v9; // rsi
  unsigned __int64 v10; // r12
  __int64 *v11; // r14
  __int64 v12; // rdx
  __int64 *v13; // r13
  __int64 *i; // rdx
  __int64 v15; // rdi
  unsigned int v16; // ecx
  __int64 v17; // rsi
  __int64 *v18; // r13
  unsigned __int64 v19; // r15
  __int64 v20; // rsi
  __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // r8
  __int64 v27; // r15
  __int64 v28; // r15
  __int64 v29; // r13
  _QWORD *v30; // rdi
  __int64 v31; // r15
  unsigned __int64 v32; // r8
  __int64 v33; // r15
  __int64 v34; // r13
  _QWORD *v35; // rdi
  unsigned int v36; // r8d
  unsigned __int64 v37; // rax
  __int64 v38; // rbx
  __int64 v41; // [rsp+18h] [rbp-A8h] BYREF
  unsigned __int64 v42[4]; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int64 v43[4]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v44[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v45; // [rsp+80h] [rbp-40h]

  v3 = (_QWORD *)sub_22077B0(0x50u);
  if ( v3 )
  {
    *v3 = v3 + 2;
    v3[1] = 0x400000000LL;
  }
  v4 = *(unsigned __int64 **)(a1 + 32);
  *(_QWORD *)(a1 + 32) = v3;
  if ( v4 )
  {
    if ( (unsigned __int64 *)*v4 != v4 + 2 )
      _libc_free(*v4);
    j_j___libc_free_0((unsigned __int64)v4);
  }
  v5 = sub_22077B0(0x190u);
  v9 = v5;
  if ( v5 )
  {
    *(_QWORD *)v5 = 0;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 24) = 0;
    *(_QWORD *)(v5 + 32) = 0;
    *(_QWORD *)(v5 + 40) = 0x800000000LL;
    *(_QWORD *)(v5 + 64) = 0x800000000LL;
    *(_QWORD *)(v5 + 200) = v5 + 216;
    *(_QWORD *)(v5 + 208) = 0x400000000LL;
    *(_QWORD *)(v5 + 248) = v5 + 264;
    *(_QWORD *)(v5 + 384) = v5 + 176;
    *(_QWORD *)(v5 + 48) = 0;
    *(_QWORD *)(v5 + 56) = 0;
    *(_QWORD *)(v5 + 72) = 0;
    *(_QWORD *)(v5 + 80) = 0;
    *(_QWORD *)(v5 + 88) = 0;
    *(_QWORD *)(v5 + 96) = 0;
    *(_QWORD *)(v5 + 104) = 0;
    *(_QWORD *)(v5 + 112) = 0;
    *(_QWORD *)(v5 + 120) = 0;
    *(_QWORD *)(v5 + 128) = 0;
    *(_QWORD *)(v5 + 136) = 0;
    *(_DWORD *)(v5 + 144) = 0;
    *(_QWORD *)(v5 + 152) = 0;
    *(_QWORD *)(v5 + 160) = 0;
    *(_QWORD *)(v5 + 168) = 0;
    *(_QWORD *)(v5 + 176) = 0;
    *(_QWORD *)(v5 + 184) = 0;
    *(_QWORD *)(v5 + 192) = 0;
    *(_QWORD *)(v5 + 256) = 0;
    *(_QWORD *)(v5 + 264) = 0;
    *(_QWORD *)(v5 + 272) = 1;
    memset((void *)(v5 + 280), 0, 0x60u);
    v6 = 0;
    *(_BYTE *)(v5 + 392) = 0;
    *(_QWORD *)(v5 + 376) = 0;
  }
  v10 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 24) = v5;
  if ( v10 )
  {
    if ( *(_DWORD *)(v10 + 376) )
      sub_EDA800(v10 + 280, (char *)sub_ED5FB0, 0, v6, v7, v8);
    v11 = *(__int64 **)(v10 + 200);
    v12 = *(unsigned int *)(v10 + 208);
    *(_QWORD *)(v10 + 176) = 0;
    v13 = &v11[v12];
    if ( v11 != v13 )
    {
      for ( i = v11; ; i = *(__int64 **)(v10 + 200) )
      {
        v15 = *v11;
        v16 = (unsigned int)(v11 - i) >> 7;
        v17 = 4096LL << v16;
        if ( v16 >= 0x1E )
          v17 = 0x40000000000LL;
        ++v11;
        sub_C7D6A0(v15, v17, 16);
        if ( v13 == v11 )
          break;
      }
    }
    v18 = *(__int64 **)(v10 + 248);
    v19 = (unsigned __int64)&v18[2 * *(unsigned int *)(v10 + 256)];
    if ( v18 != (__int64 *)v19 )
    {
      do
      {
        v20 = v18[1];
        v21 = *v18;
        v18 += 2;
        sub_C7D6A0(v21, v20, 16);
      }
      while ( (__int64 *)v19 != v18 );
      v19 = *(_QWORD *)(v10 + 248);
    }
    if ( v19 != v10 + 264 )
      _libc_free(v19);
    v22 = *(_QWORD *)(v10 + 200);
    if ( v22 != v10 + 216 )
      _libc_free(v22);
    v23 = *(_QWORD *)(v10 + 152);
    if ( v23 )
      j_j___libc_free_0(v23);
    sub_C7D6A0(*(_QWORD *)(v10 + 128), 16LL * *(unsigned int *)(v10 + 144), 8);
    v24 = *(_QWORD *)(v10 + 96);
    if ( v24 )
      j_j___libc_free_0(v24);
    v25 = *(_QWORD *)(v10 + 72);
    if ( v25 )
      j_j___libc_free_0(v25);
    v26 = *(_QWORD *)(v10 + 48);
    if ( *(_DWORD *)(v10 + 60) )
    {
      v27 = *(unsigned int *)(v10 + 56);
      if ( (_DWORD)v27 )
      {
        v28 = 8 * v27;
        v29 = 0;
        do
        {
          v30 = *(_QWORD **)(v26 + v29);
          if ( v30 != (_QWORD *)-8LL && v30 )
          {
            sub_C7D6A0((__int64)v30, *v30 + 9LL, 8);
            v26 = *(_QWORD *)(v10 + 48);
          }
          v29 += 8;
        }
        while ( v28 != v29 );
      }
    }
    _libc_free(v26);
    if ( *(_DWORD *)(v10 + 36) )
    {
      v31 = *(unsigned int *)(v10 + 32);
      v32 = *(_QWORD *)(v10 + 24);
      if ( (_DWORD)v31 )
      {
        v33 = 8 * v31;
        v34 = 0;
        do
        {
          v35 = *(_QWORD **)(v32 + v34);
          if ( v35 != (_QWORD *)-8LL && v35 )
          {
            sub_C7D6A0((__int64)v35, *v35 + 9LL, 8);
            v32 = *(_QWORD *)(v10 + 24);
          }
          v34 += 8;
        }
        while ( v34 != v33 );
      }
    }
    else
    {
      v32 = *(_QWORD *)(v10 + 24);
    }
    _libc_free(v32);
    j_j___libc_free_0(v10);
    v9 = *(_QWORD *)(a1 + 24);
  }
  sub_ED5CA0((unsigned __int64 *)&v41, v9, (__int64)a2, 1, 0, v8);
  v36 = 1;
  if ( (v41 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v37 = v41 & 0xFFFFFFFFFFFFFFFELL | 1;
    v41 = 0;
    v44[0] = v37;
    sub_C64870((__int64)v42, v44);
    sub_9C66B0(v44);
    v38 = *a2;
    sub_8FD6D0((__int64)v43, "Failed to create symtab: ", v42);
    v45 = 260;
    v44[0] = (__int64)v43;
    sub_B6ECE0(v38, (__int64)v44);
    sub_2240A30(v43);
    sub_2240A30(v42);
    sub_9C66B0(&v41);
    return 0;
  }
  return v36;
}

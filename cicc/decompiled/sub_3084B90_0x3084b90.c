// Function: sub_3084B90
// Address: 0x3084b90
//
__int64 __fastcall sub_3084B90(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rbx
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // r14
  __int64 v26; // rax
  _QWORD *v27; // rbx
  _QWORD *v28; // r15
  unsigned __int64 v29; // r8
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 v34; // [rsp+8h] [rbp-38h]
  unsigned __int64 v35; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_50208AC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_34;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_50208AC);
  v8 = *(__int64 **)(a1 + 8);
  v9 = v7 + 200;
  v10 = *v8;
  v11 = v8[1];
  if ( v10 == v11 )
LABEL_33:
    BUG();
  v12 = (__int64)&unk_501FE44;
  while ( *(_UNKNOWN **)v10 != &unk_501FE44 )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_33;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_501FE44)
      + 200;
  v14 = sub_22077B0(0x148u);
  v19 = v14;
  if ( v14 )
  {
    v20 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)v14 = a2;
    *(_QWORD *)(v14 + 8) = v9;
    *(_QWORD *)(v14 + 16) = v13;
    *(_QWORD *)(v14 + 24) = 0;
    *(_QWORD *)(v14 + 32) = 0;
    *(_QWORD *)(v14 + 40) = 0;
    *(_BYTE *)(v14 + 48) = 0;
    *(_QWORD *)(v14 + 56) = 0;
    *(_QWORD *)(v14 + 64) = 0;
    *(_QWORD *)(v14 + 72) = 0;
    *(_DWORD *)(v14 + 80) = 0;
    *(_QWORD *)(v14 + 88) = 0;
    *(_QWORD *)(v14 + 96) = 0;
    *(_QWORD *)(v14 + 104) = 0;
    *(_QWORD *)(v14 + 112) = 0;
    *(_QWORD *)(v14 + 120) = 0;
    *(_QWORD *)(v14 + 128) = 0;
    *(_DWORD *)(v14 + 136) = 0;
    *(_QWORD *)(v14 + 144) = 0;
    *(_QWORD *)(v14 + 152) = 0;
    *(_QWORD *)(v14 + 160) = 0;
    *(_DWORD *)(v14 + 168) = 0;
    v21 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v20 + 200LL))(v20);
    v22 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(v19 + 176) = v21;
    v23 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v22 + 128LL))(v22);
    *(_QWORD *)(v19 + 200) = 0;
    *(_QWORD *)(v19 + 184) = v23;
    v24 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(v19 + 208) = 0;
    *(_QWORD *)(v19 + 192) = v24;
    *(_QWORD *)(v19 + 216) = 0;
    *(_DWORD *)(v19 + 224) = 0;
    *(_QWORD *)(v19 + 232) = 0;
    *(_QWORD *)(v19 + 240) = 0;
    *(_QWORD *)(v19 + 248) = 0;
    *(_DWORD *)(v19 + 256) = 0;
    *(_QWORD *)(v19 + 264) = 0;
    *(_QWORD *)(v19 + 272) = 0;
    *(_QWORD *)(v19 + 280) = 0;
    *(_DWORD *)(v19 + 288) = 0;
    *(_QWORD *)(v19 + 296) = 0;
    *(_QWORD *)(v19 + 304) = 0;
    *(_QWORD *)(v19 + 312) = 0;
    *(_DWORD *)(v19 + 320) = 0;
  }
  v25 = *(_QWORD *)(a1 + 200);
  *(_QWORD *)(a1 + 200) = v19;
  if ( v25 )
  {
    sub_C7D6A0(*(_QWORD *)(v25 + 304), 16LL * *(unsigned int *)(v25 + 320), 8);
    sub_C7D6A0(*(_QWORD *)(v25 + 272), 4LL * *(unsigned int *)(v25 + 288), 4);
    sub_C7D6A0(*(_QWORD *)(v25 + 240), 4LL * *(unsigned int *)(v25 + 256), 4);
    sub_C7D6A0(*(_QWORD *)(v25 + 208), 4LL * *(unsigned int *)(v25 + 224), 4);
    sub_C7D6A0(*(_QWORD *)(v25 + 152), 16LL * *(unsigned int *)(v25 + 168), 8);
    v26 = *(unsigned int *)(v25 + 136);
    if ( (_DWORD)v26 )
    {
      v27 = *(_QWORD **)(v25 + 120);
      v28 = &v27[2 * v26];
      do
      {
        if ( *v27 != -8192 && *v27 != -4096 )
        {
          v29 = v27[1];
          if ( v29 )
          {
            v30 = *(_QWORD *)(v29 + 96);
            if ( v30 != v29 + 112 )
            {
              v34 = v27[1];
              _libc_free(v30);
              v29 = v34;
            }
            v31 = *(_QWORD *)(v29 + 24);
            if ( v31 != v29 + 40 )
            {
              v35 = v29;
              _libc_free(v31);
              v29 = v35;
            }
            j_j___libc_free_0(v29);
          }
        }
        v27 += 2;
      }
      while ( v28 != v27 );
      LODWORD(v26) = *(_DWORD *)(v25 + 136);
    }
    sub_C7D6A0(*(_QWORD *)(v25 + 120), 16LL * (unsigned int)v26, 8);
    v32 = *(_QWORD *)(v25 + 88);
    if ( v32 )
      j_j___libc_free_0(v32);
    sub_C7D6A0(*(_QWORD *)(v25 + 64), 8LL * *(unsigned int *)(v25 + 80), 4);
    v12 = 328;
    j_j___libc_free_0(v25);
  }
  if ( (*(_BYTE *)(**(_QWORD **)(a2 + 32) + 344LL) & 1) != 0 )
    sub_3084330(*(__int64 **)(a1 + 200), v12, v15, v16, v17, v18);
  return 0;
}

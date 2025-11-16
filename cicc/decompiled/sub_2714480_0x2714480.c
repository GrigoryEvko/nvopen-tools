// Function: sub_2714480
// Address: 0x2714480
//
__int64 __fastcall sub_2714480(__int64 a1, __int64 *a2)
{
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r11d
  _QWORD *v8; // rdx
  unsigned int v9; // ecx
  _QWORD *v10; // rax
  __int64 v11; // r9
  int v13; // eax
  int v14; // ecx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r13
  int v21; // eax
  int v22; // esi
  __int64 v23; // r8
  unsigned int v24; // eax
  __int64 v25; // rdi
  int v26; // r10d
  _QWORD *v27; // r9
  int v28; // eax
  int v29; // eax
  __int64 v30; // rdi
  int v31; // r9d
  unsigned int v32; // r14d
  _QWORD *v33; // r8
  __int64 v34; // rsi
  unsigned __int64 v35[16]; // [rsp+20h] [rbp-130h] BYREF
  __int64 v36; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v37; // [rsp+A8h] [rbp-A8h]
  __int64 v38; // [rsp+B0h] [rbp-A0h]
  _BYTE v39[8]; // [rsp+B8h] [rbp-98h] BYREF
  unsigned __int64 v40; // [rsp+C0h] [rbp-90h]
  char v41; // [rsp+D4h] [rbp-7Ch]
  _BYTE v42[16]; // [rsp+D8h] [rbp-78h] BYREF
  _BYTE v43[8]; // [rsp+E8h] [rbp-68h] BYREF
  unsigned __int64 v44; // [rsp+F0h] [rbp-60h]
  char v45; // [rsp+104h] [rbp-4Ch]
  _BYTE v46[16]; // [rsp+108h] [rbp-48h] BYREF
  char v47; // [rsp+118h] [rbp-38h]

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( v5 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = 1;
    v8 = 0;
    v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v10 = (_QWORD *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v4 == *v10 )
      return *(_QWORD *)(a1 + 32) + (v10[1] << 7) + 8LL;
    while ( v11 != -4096 )
    {
      if ( v11 == -8192 && !v8 )
        v8 = v10;
      v9 = (v5 - 1) & (v7 + v9);
      v10 = (_QWORD *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( v4 == *v10 )
        return *(_QWORD *)(a1 + 32) + (v10[1] << 7) + 8LL;
      ++v7;
    }
    if ( !v8 )
      v8 = v10;
    v13 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v14 = v13 + 1;
    if ( 4 * (v13 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 20) - v14 > v5 >> 3 )
        goto LABEL_14;
      sub_9BBF00(a1, v5);
      v28 = *(_DWORD *)(a1 + 24);
      if ( v28 )
      {
        v29 = v28 - 1;
        v30 = *(_QWORD *)(a1 + 8);
        v31 = 1;
        v32 = v29 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v33 = 0;
        v14 = *(_DWORD *)(a1 + 16) + 1;
        v8 = (_QWORD *)(v30 + 16LL * v32);
        v34 = *v8;
        if ( v4 != *v8 )
        {
          while ( v34 != -4096 )
          {
            if ( !v33 && v34 == -8192 )
              v33 = v8;
            v32 = v29 & (v31 + v32);
            v8 = (_QWORD *)(v30 + 16LL * v32);
            v34 = *v8;
            if ( v4 == *v8 )
              goto LABEL_14;
            ++v31;
          }
          if ( v33 )
            v8 = v33;
        }
        goto LABEL_14;
      }
LABEL_54:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)a1;
  }
  sub_9BBF00(a1, 2 * v5);
  v21 = *(_DWORD *)(a1 + 24);
  if ( !v21 )
    goto LABEL_54;
  v22 = v21 - 1;
  v23 = *(_QWORD *)(a1 + 8);
  v24 = (v21 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v14 = *(_DWORD *)(a1 + 16) + 1;
  v8 = (_QWORD *)(v23 + 16LL * v24);
  v25 = *v8;
  if ( v4 != *v8 )
  {
    v26 = 1;
    v27 = 0;
    while ( v25 != -4096 )
    {
      if ( !v27 && v25 == -8192 )
        v27 = v8;
      v24 = v22 & (v26 + v24);
      v8 = (_QWORD *)(v23 + 16LL * v24);
      v25 = *v8;
      if ( v4 == *v8 )
        goto LABEL_14;
      ++v26;
    }
    if ( v27 )
      v8 = v27;
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v14;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  v8[1] = 0;
  v15 = *(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32);
  v8[1] = v15 >> 7;
  memset(v35, 0, 0x78u);
  v16 = *a2;
  v35[4] = 2;
  v35[3] = (unsigned __int64)&v35[6];
  v36 = v16;
  BYTE4(v35[5]) = 1;
  v35[9] = (unsigned __int64)&v35[12];
  v35[10] = 2;
  BYTE4(v35[11]) = 1;
  v37 = 0;
  v38 = 0;
  sub_C8CF70((__int64)v39, v42, 2, (__int64)&v35[6], (__int64)&v35[2]);
  sub_C8CF70((__int64)v43, v46, 2, (__int64)&v35[12], (__int64)&v35[8]);
  v20 = *(_QWORD *)(a1 + 40);
  v47 = v35[14];
  if ( v20 == *(_QWORD *)(a1 + 48) )
  {
    sub_2712560((unsigned __int64 *)(a1 + 32), v20, (__int64)&v36, v17, v18, v19);
  }
  else
  {
    if ( v20 )
    {
      *(_QWORD *)v20 = v36;
      *(_WORD *)(v20 + 8) = v37;
      *(_QWORD *)(v20 + 16) = v38;
      sub_C8CF70(v20 + 24, (void *)(v20 + 56), 2, (__int64)v42, (__int64)v39);
      sub_C8CF70(v20 + 72, (void *)(v20 + 104), 2, (__int64)v46, (__int64)v43);
      *(_BYTE *)(v20 + 120) = v47;
      v20 = *(_QWORD *)(a1 + 40);
    }
    *(_QWORD *)(a1 + 40) = v20 + 128;
  }
  if ( v45 )
  {
    if ( v41 )
      goto LABEL_22;
  }
  else
  {
    _libc_free(v44);
    if ( v41 )
    {
LABEL_22:
      if ( BYTE4(v35[11]) )
        goto LABEL_23;
LABEL_27:
      _libc_free(v35[9]);
      if ( BYTE4(v35[5]) )
        return *(_QWORD *)(a1 + 32) + v15 + 8;
LABEL_28:
      _libc_free(v35[3]);
      return *(_QWORD *)(a1 + 32) + v15 + 8;
    }
  }
  _libc_free(v40);
  if ( !BYTE4(v35[11]) )
    goto LABEL_27;
LABEL_23:
  if ( !BYTE4(v35[5]) )
    goto LABEL_28;
  return *(_QWORD *)(a1 + 32) + v15 + 8;
}

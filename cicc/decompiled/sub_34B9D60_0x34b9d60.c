// Function: sub_34B9D60
// Address: 0x34b9d60
//
void __fastcall sub_34B9D60(__int64 a1, int a2, __int64 a3)
{
  _QWORD *v5; // rdi
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rbx
  unsigned int v9; // esi
  __int64 v10; // r10
  int v11; // r15d
  __int64 v12; // rcx
  __int64 v13; // r9
  __int64 *v14; // rdx
  __int64 v15; // r8
  int v16; // eax
  int v17; // edx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdi
  int v20; // ecx
  __int64 j; // rdi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // r15
  int v27; // ebx
  unsigned __int64 v28; // rdx
  _QWORD *v29; // rcx
  unsigned __int64 i; // rax
  int v31; // eax
  int v32; // esi
  __int64 v33; // rdi
  unsigned int v34; // eax
  int v35; // r10d
  int v36; // eax
  int v37; // eax
  __int64 v38; // rdi
  unsigned int v39; // r13d
  __int64 v40; // rsi
  _QWORD *v42; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v43; // [rsp+28h] [rbp-B8h]
  _QWORD v44[22]; // [rsp+30h] [rbp-B0h] BYREF

  v5 = v44;
  v43 = 0x1000000001LL;
  v6 = 1;
  v42 = v44;
  v44[0] = a3;
  do
  {
    while ( 1 )
    {
      v7 = v6--;
      v8 = v5[v7 - 1];
      LODWORD(v43) = v6;
      if ( *(_BYTE *)(v8 + 216) && a3 != v8 )
        goto LABEL_3;
      v9 = *(_DWORD *)(a1 + 24);
      if ( !v9 )
      {
        ++*(_QWORD *)a1;
LABEL_38:
        sub_2E3ADF0(a1, 2 * v9);
        v31 = *(_DWORD *)(a1 + 24);
        if ( !v31 )
          goto LABEL_67;
        v32 = v31 - 1;
        v33 = *(_QWORD *)(a1 + 8);
        v34 = (v31 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v17 = *(_DWORD *)(a1 + 16) + 1;
        v12 = v33 + 16LL * v34;
        v15 = *(_QWORD *)v12;
        if ( v8 != *(_QWORD *)v12 )
        {
          v35 = 1;
          v13 = 0;
          while ( v15 != -4096 )
          {
            if ( v15 == -8192 && !v13 )
              v13 = v12;
            v34 = v32 & (v35 + v34);
            v12 = v33 + 16LL * v34;
            v15 = *(_QWORD *)v12;
            if ( v8 == *(_QWORD *)v12 )
              goto LABEL_12;
            ++v35;
          }
          if ( v13 )
            v12 = v13;
        }
        goto LABEL_12;
      }
      v10 = *(_QWORD *)(a1 + 8);
      v11 = 1;
      v12 = 0;
      v13 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v14 = (__int64 *)(v10 + 16 * v13);
      v15 = *v14;
      if ( v8 != *v14 )
        break;
LABEL_3:
      if ( !v6 )
        goto LABEL_25;
    }
    while ( v15 != -4096 )
    {
      if ( v12 || v15 != -8192 )
        v14 = (__int64 *)v12;
      v13 = (v9 - 1) & (v11 + (_DWORD)v13);
      v15 = *(_QWORD *)(v10 + 16LL * (unsigned int)v13);
      if ( v8 == v15 )
        goto LABEL_3;
      ++v11;
      v12 = (__int64)v14;
      v14 = (__int64 *)(v10 + 16LL * (unsigned int)v13);
    }
    v16 = *(_DWORD *)(a1 + 16);
    if ( !v12 )
      v12 = (__int64)v14;
    ++*(_QWORD *)a1;
    v17 = v16 + 1;
    if ( 4 * (v16 + 1) >= 3 * v9 )
      goto LABEL_38;
    if ( v9 - *(_DWORD *)(a1 + 20) - v17 <= v9 >> 3 )
    {
      sub_2E3ADF0(a1, v9);
      v36 = *(_DWORD *)(a1 + 24);
      if ( !v36 )
      {
LABEL_67:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 8);
      v15 = 0;
      v39 = v37 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v13 = 1;
      v17 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v38 + 16LL * v39;
      v40 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        while ( v40 != -4096 )
        {
          if ( !v15 && v40 == -8192 )
            v15 = v12;
          v39 = v37 & (v13 + v39);
          v12 = v38 + 16LL * v39;
          v40 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_12;
          v13 = (unsigned int)(v13 + 1);
        }
        if ( v15 )
          v12 = v15;
      }
    }
LABEL_12:
    *(_DWORD *)(a1 + 16) = v17;
    if ( *(_QWORD *)v12 != -4096 )
      --*(_DWORD *)(a1 + 20);
    *(_QWORD *)v12 = v8;
    *(_DWORD *)(v12 + 8) = a2;
    v18 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    v19 = v18;
    if ( v18 == v8 + 48 )
    {
LABEL_28:
      v23 = *(unsigned int *)(v8 + 120);
      v24 = (unsigned int)v43;
      v25 = *(_QWORD *)(v8 + 112);
      v26 = 8 * v23;
      v27 = *(_DWORD *)(v8 + 120);
      v28 = (unsigned int)v43 + v23;
      if ( v28 > HIDWORD(v43) )
      {
        sub_C8D5F0((__int64)&v42, v44, v28, 8u, v15, v13);
        v24 = (unsigned int)v43;
      }
      v5 = v42;
      v29 = &v42[v24];
      if ( v26 )
      {
        for ( i = 0; i != v26; i += 8LL )
          v29[i / 8] = *(_QWORD *)(v25 + i);
        LODWORD(v24) = v43;
        v5 = v42;
      }
      LODWORD(v43) = v27 + v24;
      v6 = v27 + v24;
      goto LABEL_3;
    }
    if ( !v18 )
      BUG();
    v20 = *(_DWORD *)(v18 + 44);
    if ( (*(_QWORD *)v18 & 4) != 0 )
    {
      if ( (v20 & 4) != 0 )
        goto LABEL_36;
    }
    else if ( (v20 & 4) != 0 )
    {
      for ( j = *(_QWORD *)v18; ; j = *(_QWORD *)v19 )
      {
        v19 = j & 0xFFFFFFFFFFFFFFF8LL;
        v20 = *(_DWORD *)(v19 + 44) & 0xFFFFFF;
        if ( (*(_DWORD *)(v19 + 44) & 4) == 0 )
          break;
      }
    }
    if ( (v20 & 8) != 0 )
    {
      LOBYTE(v22) = sub_2E88A90(v19, 64, 1);
      goto LABEL_23;
    }
LABEL_36:
    v22 = (*(_QWORD *)(*(_QWORD *)(v19 + 16) + 24LL) >> 6) & 1LL;
LABEL_23:
    if ( !(_BYTE)v22 )
      goto LABEL_28;
    v6 = v43;
    v5 = v42;
  }
  while ( (_DWORD)v43 );
LABEL_25:
  if ( v5 != v44 )
    _libc_free((unsigned __int64)v5);
}

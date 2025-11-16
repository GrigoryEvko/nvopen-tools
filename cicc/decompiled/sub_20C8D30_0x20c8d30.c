// Function: sub_20C8D30
// Address: 0x20c8d30
//
void __fastcall sub_20C8D30(__int64 a1, int a2, __int64 a3)
{
  _QWORD *v5; // rdi
  __int64 i; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  unsigned int v9; // esi
  __int64 v10; // r10
  int v11; // r15d
  _QWORD *v12; // rcx
  _QWORD *v13; // r9
  _QWORD *v14; // rdx
  _QWORD *v15; // r8
  int v16; // eax
  int v17; // edx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx
  __int16 v21; // ax
  __int64 j; // rdi
  __int64 v23; // rax
  __int64 *v24; // r13
  __int64 *v25; // rbx
  __int64 v26; // r15
  int v27; // eax
  int v28; // edi
  __int64 v29; // rsi
  unsigned int v30; // eax
  int v31; // r10d
  int v32; // eax
  int v33; // eax
  __int64 v34; // rdi
  unsigned int v35; // r13d
  __int64 v36; // rsi
  _QWORD *v38; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v39; // [rsp+28h] [rbp-B8h]
  _QWORD v40[22]; // [rsp+30h] [rbp-B0h] BYREF

  v5 = v40;
  v39 = 0x1000000001LL;
  LODWORD(i) = 1;
  v38 = v40;
  v40[0] = a3;
  do
  {
    while ( 1 )
    {
      v7 = (unsigned int)i;
      LODWORD(i) = i - 1;
      v8 = v5[v7 - 1];
      LODWORD(v39) = i;
      if ( *(_BYTE *)(v8 + 180) && a3 != v8 )
        goto LABEL_3;
      v9 = *(_DWORD *)(a1 + 24);
      if ( !v9 )
      {
        ++*(_QWORD *)a1;
LABEL_36:
        sub_1DDD540(a1, 2 * v9);
        v27 = *(_DWORD *)(a1 + 24);
        if ( !v27 )
          goto LABEL_65;
        v28 = v27 - 1;
        v29 = *(_QWORD *)(a1 + 8);
        v30 = (v27 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v17 = *(_DWORD *)(a1 + 16) + 1;
        v12 = (_QWORD *)(v29 + 16LL * v30);
        v15 = (_QWORD *)*v12;
        if ( v8 != *v12 )
        {
          v31 = 1;
          v13 = 0;
          while ( v15 != (_QWORD *)-8LL )
          {
            if ( !v13 && v15 == (_QWORD *)-16LL )
              v13 = v12;
            v30 = v28 & (v31 + v30);
            v12 = (_QWORD *)(v29 + 16LL * v30);
            v15 = (_QWORD *)*v12;
            if ( v8 == *v12 )
              goto LABEL_12;
            ++v31;
          }
          if ( v13 )
            v12 = v13;
        }
        goto LABEL_12;
      }
      v10 = *(_QWORD *)(a1 + 8);
      v11 = 1;
      v12 = 0;
      LODWORD(v13) = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v14 = (_QWORD *)(v10 + 16LL * (unsigned int)v13);
      v15 = (_QWORD *)*v14;
      if ( v8 != *v14 )
        break;
LABEL_3:
      if ( !(_DWORD)i )
        goto LABEL_25;
    }
    while ( v15 != (_QWORD *)-8LL )
    {
      if ( v12 || v15 != (_QWORD *)-16LL )
        v14 = v12;
      LODWORD(v13) = (v9 - 1) & (v11 + (_DWORD)v13);
      v15 = *(_QWORD **)(v10 + 16LL * (unsigned int)v13);
      if ( (_QWORD *)v8 == v15 )
        goto LABEL_3;
      ++v11;
      v12 = v14;
      v14 = (_QWORD *)(v10 + 16LL * (unsigned int)v13);
    }
    v16 = *(_DWORD *)(a1 + 16);
    if ( !v12 )
      v12 = v14;
    ++*(_QWORD *)a1;
    v17 = v16 + 1;
    if ( 4 * (v16 + 1) >= 3 * v9 )
      goto LABEL_36;
    if ( v9 - *(_DWORD *)(a1 + 20) - v17 <= v9 >> 3 )
    {
      sub_1DDD540(a1, v9);
      v32 = *(_DWORD *)(a1 + 24);
      if ( !v32 )
      {
LABEL_65:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 8);
      v15 = 0;
      v35 = v33 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      LODWORD(v13) = 1;
      v17 = *(_DWORD *)(a1 + 16) + 1;
      v12 = (_QWORD *)(v34 + 16LL * v35);
      v36 = *v12;
      if ( v8 != *v12 )
      {
        while ( v36 != -8 )
        {
          if ( v36 == -16 && !v15 )
            v15 = v12;
          v35 = v33 & ((_DWORD)v13 + v35);
          v12 = (_QWORD *)(v34 + 16LL * v35);
          v36 = *v12;
          if ( v8 == *v12 )
            goto LABEL_12;
          LODWORD(v13) = (_DWORD)v13 + 1;
        }
        if ( v15 )
          v12 = v15;
      }
    }
LABEL_12:
    *(_DWORD *)(a1 + 16) = v17;
    if ( *v12 != -8 )
      --*(_DWORD *)(a1 + 20);
    *v12 = v8;
    *((_DWORD *)v12 + 2) = a2;
    v18 = *(_QWORD *)(v8 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    v19 = v18;
    if ( v18 == v8 + 24 )
    {
LABEL_28:
      v24 = *(__int64 **)(v8 + 96);
      v25 = *(__int64 **)(v8 + 88);
      for ( i = (unsigned int)v39; v24 != v25; LODWORD(v39) = v39 + 1 )
      {
        v26 = *v25;
        if ( (unsigned int)i >= HIDWORD(v39) )
        {
          sub_16CD150((__int64)&v38, v40, 0, 8, (int)v15, (int)v13);
          i = (unsigned int)v39;
        }
        ++v25;
        v38[i] = v26;
        i = (unsigned int)(v39 + 1);
      }
      v5 = v38;
      goto LABEL_3;
    }
    if ( !v18 )
      BUG();
    v20 = *(_QWORD *)v18;
    v21 = *(_WORD *)(v18 + 46);
    if ( (v20 & 4) != 0 )
    {
      if ( (v21 & 4) != 0 )
        goto LABEL_34;
    }
    else if ( (v21 & 4) != 0 )
    {
      for ( j = v20; ; j = *(_QWORD *)v19 )
      {
        v19 = j & 0xFFFFFFFFFFFFFFF8LL;
        v21 = *(_WORD *)(v19 + 46);
        if ( (v21 & 4) == 0 )
          break;
      }
    }
    if ( (v21 & 8) != 0 )
    {
      LOBYTE(v23) = sub_1E15D00(v19, 8u, 1);
      goto LABEL_23;
    }
LABEL_34:
    v23 = (*(_QWORD *)(*(_QWORD *)(v19 + 16) + 8LL) >> 3) & 1LL;
LABEL_23:
    if ( !(_BYTE)v23 )
      goto LABEL_28;
    LODWORD(i) = v39;
    v5 = v38;
  }
  while ( (_DWORD)v39 );
LABEL_25:
  if ( v5 != v40 )
    _libc_free((unsigned __int64)v5);
}

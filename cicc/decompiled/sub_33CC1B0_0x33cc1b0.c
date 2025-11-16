// Function: sub_33CC1B0
// Address: 0x33cc1b0
//
void __fastcall sub_33CC1B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // r15
  unsigned __int64 v9; // r13
  unsigned int v10; // eax
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  unsigned __int64 *v13; // rcx
  unsigned __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rsi
  int v17; // edx
  unsigned int v18; // eax
  __int64 *v19; // r13
  __int64 v20; // rcx
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // r8
  unsigned int v23; // r14d
  _QWORD *v24; // rax
  _QWORD *v25; // rdx
  int v26; // edi
  unsigned __int64 v27; // [rsp+8h] [rbp-38h]

  v8 = *(_QWORD **)(a2 + 40);
  if ( v8 )
  {
    v9 = *(unsigned int *)(a2 + 64);
    v10 = 0;
    if ( *(_DWORD *)(a2 + 64) && (--v9, v9) )
    {
      _BitScanReverse64(&v9, v9);
      v11 = *(unsigned int *)(a1 + 648);
      v10 = 64 - (v9 ^ 0x3F);
      v9 = 8LL * (int)v10;
      if ( (unsigned int)v11 > v10 )
        goto LABEL_4;
    }
    else
    {
      v11 = *(unsigned int *)(a1 + 648);
      if ( (_DWORD)v11 )
        goto LABEL_4;
    }
    v22 = v10 + 1;
    v23 = v10 + 1;
    if ( v22 != v11 )
    {
      if ( v22 >= v11 )
      {
        if ( v22 > *(unsigned int *)(a1 + 652) )
        {
          v27 = v10 + 1;
          sub_C8D5F0(a1 + 640, (const void *)(a1 + 656), v22, 8u, v22, a6);
          v11 = *(unsigned int *)(a1 + 648);
          v22 = v27;
        }
        v12 = *(_QWORD *)(a1 + 640);
        v24 = (_QWORD *)(v12 + 8 * v11);
        v25 = (_QWORD *)(v12 + 8 * v22);
        if ( v24 != v25 )
        {
          do
          {
            if ( v24 )
              *v24 = 0;
            ++v24;
          }
          while ( v25 != v24 );
          v12 = *(_QWORD *)(a1 + 640);
        }
        *(_DWORD *)(a1 + 648) = v23;
        goto LABEL_5;
      }
      *(_DWORD *)(a1 + 648) = v22;
    }
LABEL_4:
    v12 = *(_QWORD *)(a1 + 640);
LABEL_5:
    *v8 = *(_QWORD *)(v12 + v9);
    *(_QWORD *)(*(_QWORD *)(a1 + 640) + v9) = v8;
    *(_DWORD *)(a2 + 64) = 0;
    *(_QWORD *)(a2 + 40) = 0;
  }
  v13 = *(unsigned __int64 **)(a2 + 16);
  v14 = *(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  *v13 = v14 | *v13 & 7;
  *(_QWORD *)(v14 + 8) = v13;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 8) &= 7uLL;
  *(_QWORD *)a2 = *(_QWORD *)(a1 + 416);
  *(_QWORD *)(a1 + 416) = a2;
  *(_DWORD *)(a2 + 24) = 0;
  sub_33CC0C0(*(_QWORD *)(a1 + 720), a2);
  v15 = *(_DWORD *)(a1 + 752);
  v16 = *(_QWORD *)(a1 + 736);
  if ( v15 )
  {
    v17 = v15 - 1;
    v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v19 = (__int64 *)(v16 + 80LL * v18);
    v20 = *v19;
    if ( *v19 == a2 )
    {
LABEL_8:
      v21 = v19[1];
      if ( (__int64 *)v21 != v19 + 3 )
        _libc_free(v21);
      *v19 = -8192;
      --*(_DWORD *)(a1 + 744);
      ++*(_DWORD *)(a1 + 748);
    }
    else
    {
      v26 = 1;
      while ( v20 != -4096 )
      {
        v18 = v17 & (v26 + v18);
        v19 = (__int64 *)(v16 + 80LL * v18);
        v20 = *v19;
        if ( *v19 == a2 )
          goto LABEL_8;
        ++v26;
      }
    }
  }
}

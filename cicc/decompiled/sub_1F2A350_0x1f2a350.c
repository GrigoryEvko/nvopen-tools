// Function: sub_1F2A350
// Address: 0x1f2a350
//
__int64 __fastcall sub_1F2A350(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 (*v7)(void); // rdx
  __int64 v8; // rax
  unsigned int v9; // r15d
  unsigned int v11; // ebx
  __int64 v12; // rdx
  void *v13; // rax
  __int64 v14; // r13
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r13
  __int64 i; // r13
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 k; // r13
  _DWORD *v21; // rax
  unsigned int *v22; // rdx
  _DWORD *v23; // r15
  unsigned int *j; // rsi
  unsigned int v25; // ecx
  __int64 v26; // rdi
  void (*v27)(); // rax
  unsigned __int8 v28; // [rsp+17h] [rbp-79h]
  _QWORD *v30; // [rsp+20h] [rbp-70h]
  __int64 v31; // [rsp+28h] [rbp-68h]
  __m128i v32; // [rsp+30h] [rbp-60h] BYREF
  __int64 v33; // [rsp+40h] [rbp-50h]
  _DWORD *v34; // [rsp+48h] [rbp-48h]

  if ( !byte_4FCAB80 )
    return 0;
  v7 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 112LL);
  v8 = 0;
  if ( v7 != sub_1D00B10 )
    v8 = v7();
  *(_QWORD *)(a1 + 232) = v8;
  v28 = *(_BYTE *)(*(_QWORD *)(a2 + 56) + 40LL);
  if ( v28 && (v30 = *(_QWORD **)(a2 + 328), v30 != (_QWORD *)(a2 + 320)) )
  {
    v9 = 0;
    while ( 1 )
    {
      *(_QWORD *)(a1 + 240) = v8;
      *(_DWORD *)(a1 + 256) = 0;
      v11 = *(_DWORD *)(v8 + 16);
      v12 = *(_DWORD *)(a1 + 304) >> 2;
      if ( v11 < (unsigned int)v12 || v11 > *(_DWORD *)(a1 + 304) )
      {
        _libc_free(*(_QWORD *)(a1 + 296));
        v13 = _libc_calloc(v11, 1u);
        v12 = v11;
        v14 = (__int64)v13;
        if ( !v13 && (v11 || (v14 = malloc(1u)) == 0) )
          sub_16BD1C0("Allocation failed", 1u);
        *(_QWORD *)(a1 + 296) = v14;
        *(_DWORD *)(a1 + 304) = v11;
      }
      sub_1DC29C0((__int64 *)(a1 + 240), v30, v12, a4, a5, a6);
      v31 = v30[3];
      v15 = v31 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v31 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        BUG();
      v16 = v31 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v15 & 4) == 0 && (*(_BYTE *)(v15 + 46) & 4) != 0 )
      {
        for ( i = *(_QWORD *)v15; ; i = *(_QWORD *)v16 )
        {
          v16 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v16 + 46) & 4) == 0 )
            break;
        }
      }
      while ( v30 + 3 != (_QWORD *)v16 )
      {
        if ( **(_WORD **)(v16 + 16) == 21 )
        {
          v21 = sub_1E0B760(a2);
          v22 = *(unsigned int **)(a1 + 248);
          v23 = v21;
          for ( j = &v22[*(unsigned int *)(a1 + 256)]; v22 != j; v21[v25 >> 5] |= 1 << v25 )
            v25 = *v22++;
          v26 = *(_QWORD *)(a1 + 232);
          v27 = *(void (**)())(*(_QWORD *)v26 + 88LL);
          if ( v27 != nullsub_755 )
            ((void (__fastcall *)(__int64, _DWORD *))v27)(v26, v23);
          v34 = v23;
          v32.m128i_i64[0] = 13;
          v33 = 0;
          sub_1E1A9C0(v16, a2, &v32);
          v9 = v28;
        }
        sub_1DC2260((__int64 *)(a1 + 240), v16, v15, a4, a5, a6);
        v18 = (__int64 *)(*(_QWORD *)v16 & 0xFFFFFFFFFFFFFFF8LL);
        v15 = (unsigned __int64)v18;
        if ( !v18 )
          BUG();
        v16 = *(_QWORD *)v16 & 0xFFFFFFFFFFFFFFF8LL;
        v19 = *v18;
        if ( (v19 & 4) == 0 && (*(_BYTE *)(v15 + 46) & 4) != 0 )
        {
          for ( k = v19; ; k = *(_QWORD *)v16 )
          {
            v16 = k & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v16 + 46) & 4) == 0 )
              break;
          }
        }
      }
      v30 = (_QWORD *)v30[1];
      if ( (_QWORD *)(a2 + 320) == v30 )
        break;
      v8 = *(_QWORD *)(a1 + 232);
    }
  }
  else
  {
    return 0;
  }
  return v9;
}

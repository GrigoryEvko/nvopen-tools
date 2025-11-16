// Function: sub_1DF4550
// Address: 0x1df4550
//
__int64 __fastcall sub_1DF4550(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 v7; // rdi
  unsigned int v8; // r8d
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned int v12; // edi
  int *v13; // rcx
  int v14; // r8d
  __int64 v15; // r14
  __int64 v16; // rcx
  unsigned int v17; // r10d
  __int64 v18; // r11
  unsigned int v19; // r9d
  __int16 *v20; // rdi
  __int16 v21; // ax
  __int16 *v22; // rdi
  unsigned __int16 v23; // si
  __int16 *v24; // rax
  __int16 v25; // cx
  int v26; // r15d
  unsigned int i; // r15d
  int v28; // ecx
  int v29; // r10d
  unsigned int v30; // [rsp+4h] [rbp-3Ch]
  __int64 v31; // [rsp+8h] [rbp-38h]

  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 248) + 304LL);
  if ( (*(_QWORD *)(v7 + 8LL * (a3 >> 6)) & (1LL << a3)) != 0 )
    return 0;
  if ( (*(_QWORD *)(v7 + 8LL * (a4 >> 6)) & (1LL << a4)) != 0 )
    return 0;
  v10 = *(unsigned int *)(a1 + 440);
  if ( !(_DWORD)v10 )
    return 0;
  v11 = *(_QWORD *)(a1 + 424);
  v12 = (v10 - 1) & (37 * a4);
  v13 = (int *)(v11 + 16LL * v12);
  v14 = *v13;
  if ( *v13 != a4 )
  {
    v28 = 1;
    while ( v14 != -1 )
    {
      v29 = v28 + 1;
      v12 = (v10 - 1) & (v28 + v12);
      v13 = (int *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( *v13 == a4 )
        goto LABEL_7;
      v28 = v29;
    }
    return 0;
  }
LABEL_7:
  if ( v13 == (int *)(v11 + 16 * v10) )
    return 0;
  v15 = *((_QWORD *)v13 + 1);
  v16 = *(_QWORD *)(v15 + 32);
  v8 = *(unsigned __int8 *)(v16 + 3);
  LOBYTE(v8) = ((v8 & 0x10) != 0) & ((unsigned __int8)v8 >> 6);
  if ( (_BYTE)v8 )
    return 0;
  v17 = *(_DWORD *)(v16 + 48);
  if ( a3 == v17 )
  {
LABEL_18:
    for ( i = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL); a2 != v15; v15 = *(_QWORD *)(v15 + 8) )
      sub_1E1A450(v15, i, *(_QWORD *)(a1 + 232));
    sub_1E16240(a2);
    *(_BYTE *)(a1 + 512) = 1;
    return 1;
  }
  v18 = *(_QWORD *)(a1 + 232);
  v19 = *(_DWORD *)(v16 + 8);
  v20 = (__int16 *)(*(_QWORD *)(v18 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v18 + 8) + 24LL * a3 + 8));
  v21 = *v20;
  v22 = v20 + 1;
  v23 = v21 + a3;
  if ( !v21 )
    v22 = 0;
LABEL_12:
  v24 = v22;
  if ( v22 )
  {
    while ( v17 != v23 )
    {
      v25 = *v24;
      v22 = 0;
      ++v24;
      if ( !v25 )
        goto LABEL_12;
      v23 += v25;
      if ( !v24 )
        return v8;
    }
    v30 = v19;
    v31 = v18 + 8;
    v26 = sub_38D7050(v18 + 8, v17);
    if ( v26 == (unsigned int)sub_38D7050(v31, v30) )
      goto LABEL_18;
    return 0;
  }
  return v8;
}

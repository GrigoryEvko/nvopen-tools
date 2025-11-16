// Function: sub_37BB7D0
// Address: 0x37bb7d0
//
__int64 __fastcall sub_37BB7D0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, _QWORD *a5, __int64 a6)
{
  __int64 v6; // r15
  unsigned __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // r8
  _BYTE *v11; // r14
  int v12; // eax
  _BYTE *v13; // rcx
  __int64 v14; // rax
  unsigned int v15; // eax
  __int64 v16; // r13
  _BYTE *v17; // r12
  unsigned __int64 v18; // rax
  _QWORD *v19; // r14
  _QWORD *v20; // r15
  __int64 v21; // r12
  _QWORD *v22; // rsi
  unsigned int v23; // r13d
  int v24; // r9d
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r11
  unsigned int v28; // r11d
  unsigned int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // r10
  unsigned int v32; // eax
  __int64 v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 i; // rcx
  __int64 v37; // rdi
  _QWORD *v38; // r13
  _BYTE *v39; // rax
  char v40; // r10
  __int64 v41; // rdx
  int v43; // eax
  int v44; // eax
  int v45; // r10d
  unsigned __int8 v46; // [rsp+14h] [rbp-9Ch]
  int v47; // [rsp+14h] [rbp-9Ch]
  __int64 v49; // [rsp+18h] [rbp-98h]
  _BYTE *v52; // [rsp+30h] [rbp-80h] BYREF
  __int64 v53; // [rsp+38h] [rbp-78h]
  _BYTE v54[112]; // [rsp+40h] [rbp-70h] BYREF

  v6 = a2;
  v8 = *(unsigned int *)(a2 + 72);
  v52 = v54;
  v9 = *(_QWORD *)(a2 + 64);
  v10 = 8 * v8;
  v53 = 0x800000000LL;
  if ( v8 > 8 )
  {
    sub_C8D5F0((__int64)&v52, v54, v8, 8u, v10, a6);
    v11 = v52;
    v12 = v53;
    v10 = 8 * v8;
    v13 = &v52[8 * (unsigned int)v53];
  }
  else
  {
    v11 = v54;
    v12 = 0;
    v13 = v54;
  }
  if ( v10 )
  {
    v14 = 0;
    do
    {
      *(_QWORD *)&v13[8 * v14] = *(_QWORD *)(v9 + 8 * v14);
      ++v14;
    }
    while ( (__int64)(v8 - v14) > 0 );
    v11 = v52;
    v12 = v53;
  }
  v15 = v12 + v8;
  LODWORD(v53) = v15;
  v16 = 8LL * v15;
  v17 = &v11[v16];
  if ( v11 == &v11[v16] )
    goto LABEL_21;
  _BitScanReverse64(&v18, v16 >> 3);
  sub_37B8650((__int64 *)v11, &v11[v16], 2LL * (int)(63 - (v18 ^ 0x3F)), a1);
  if ( (unsigned __int64)v16 <= 0x80 )
  {
    sub_37B8B00(v11, &v11[v16], a1);
    goto LABEL_40;
  }
  sub_37B8B00(v11, v11 + 128, a1);
  v19 = v11 + 128;
  if ( v17 == (_BYTE *)v19 )
  {
LABEL_40:
    v15 = v53;
    v17 = v52;
    goto LABEL_21;
  }
  v20 = v17;
  do
  {
    v21 = *v19;
    v22 = v19;
    v23 = ((unsigned int)*v19 >> 9) ^ ((unsigned int)*v19 >> 4);
    while ( 1 )
    {
      v33 = *(unsigned int *)(a1 + 688);
      v34 = *(v22 - 1);
      v35 = *(_QWORD *)(a1 + 672);
      if ( !(_DWORD)v33 )
        break;
      v24 = v33 - 1;
      v25 = (v33 - 1) & v23;
      v26 = (__int64 *)(v35 + 16LL * v25);
      v27 = *v26;
      if ( v21 == *v26 )
      {
LABEL_13:
        v28 = *((_DWORD *)v26 + 2);
      }
      else
      {
        v44 = 1;
        while ( v27 != -4096 )
        {
          v45 = v44 + 1;
          v25 = v24 & (v44 + v25);
          v26 = (__int64 *)(v35 + 16LL * v25);
          v27 = *v26;
          if ( v21 == *v26 )
            goto LABEL_13;
          v44 = v45;
        }
        v28 = *(_DWORD *)(v35 + 16LL * (unsigned int)v33 + 8);
      }
      v29 = v24 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v30 = (__int64 *)(v35 + 16LL * v29);
      v31 = *v30;
      if ( v34 == *v30 )
      {
LABEL_15:
        v32 = *((_DWORD *)v30 + 2);
      }
      else
      {
        v43 = 1;
        while ( v31 != -4096 )
        {
          v29 = v24 & (v43 + v29);
          v47 = v43 + 1;
          v30 = (__int64 *)(v35 + 16LL * v29);
          v31 = *v30;
          if ( v34 == *v30 )
            goto LABEL_15;
          v43 = v47;
        }
        v32 = *(_DWORD *)(v35 + 16 * v33 + 8);
      }
      if ( v28 >= v32 )
        break;
      *v22-- = v34;
    }
    ++v19;
    *v22 = v21;
  }
  while ( v20 != v19 );
  v6 = a2;
  v15 = v53;
  v17 = v52;
LABEL_21:
  v46 = 0;
  if ( v15 && *(_DWORD *)(*(_QWORD *)(a1 + 408) + 40LL) )
  {
    v49 = *(unsigned int *)(*(_QWORD *)(a1 + 408) + 40LL);
    for ( i = 0; v49 != i; ++i )
    {
      while ( 1 )
      {
        v37 = *(_QWORD *)(**(_QWORD **)(*a4 + 8LL * *(int *)(*(_QWORD *)v17 + 24LL)) + 8 * i);
        v38 = (_QWORD *)(8 * i + *a5);
        if ( ((i << 40) | *(_DWORD *)(v6 + 24) & 0xFFFFF) == *v38 )
          break;
        if ( v37 != *v38 )
          goto LABEL_25;
LABEL_26:
        if ( v49 == ++i )
          goto LABEL_36;
      }
      if ( (unsigned int)v53 <= 1 )
        goto LABEL_25;
      v39 = v17 + 8;
      v40 = 0;
      do
      {
        v41 = *(_QWORD *)(**(_QWORD **)(*a4 + 8LL * *(int *)(*(_QWORD *)v39 + 24LL)) + 8 * i);
        if ( v37 != v41 && v41 != ((i << 40) | *(_DWORD *)(v6 + 24) & 0xFFFFF) )
          v40 = 1;
        v39 += 8;
      }
      while ( &v17[8 * (unsigned int)(v53 - 2) + 16] != v39 );
      if ( !v40 )
      {
LABEL_25:
        v46 = 1;
        *v38 = v37;
        v17 = v52;
        goto LABEL_26;
      }
    }
  }
LABEL_36:
  if ( v17 != v54 )
    _libc_free((unsigned __int64)v17);
  return v46;
}

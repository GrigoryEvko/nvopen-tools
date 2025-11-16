// Function: sub_16A5530
// Address: 0x16a5530
//
__int64 __fastcall sub_16A5530(__int64 a1, __int64 *a2, unsigned int a3, unsigned int a4)
{
  unsigned int v7; // edx
  unsigned int v8; // r14d
  unsigned int v9; // eax
  char v10; // r13
  unsigned __int64 v12; // r9
  unsigned __int64 v13; // r11
  unsigned __int64 *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // r10
  __int64 v17; // r15
  unsigned __int64 *v18; // r14
  unsigned int v19; // esi
  __int64 v20; // rdi
  unsigned int v21; // eax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  unsigned int v24; // eax
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+18h] [rbp-38h]

  v7 = *((_DWORD *)a2 + 2);
  if ( v7 <= 0x40 )
  {
    v25 = *a2;
    *(_DWORD *)(a1 + 8) = a3;
    v26 = v25 >> a4;
    if ( a3 > 0x40 )
    {
LABEL_22:
      sub_16A4EF0(a1, v26, 0);
      return a1;
    }
LABEL_20:
    *(_QWORD *)a1 = (0xFFFFFFFFFFFFFFFFLL >> -(char)a3) & v26;
    return a1;
  }
  v8 = a4 >> 6;
  v9 = (a3 + a4 - 1) >> 6;
  v10 = a4 & 0x3F;
  if ( a4 >> 6 == v9 )
  {
    v27 = *(_QWORD *)(*a2 + 8LL * v8);
    *(_DWORD *)(a1 + 8) = a3;
    v26 = v27 >> v10;
    if ( a3 > 0x40 )
      goto LABEL_22;
    goto LABEL_20;
  }
  if ( (a4 & 0x3F) == 0 )
  {
    sub_16A50F0(a1, a3, (_QWORD *)(*a2 + 8LL * v8), v9 - v8 + 1);
    return a1;
  }
  v30 = a3;
  if ( a3 <= 0x40 )
  {
    v29 = 0;
    v13 = (unsigned __int64)(a3 + 63) >> 6;
    v12 = ((unsigned __int64)v7 + 63) >> 6;
    goto LABEL_24;
  }
  sub_16A4EF0((__int64)&v29, 0, 0);
  v12 = ((unsigned __int64)*((unsigned int *)a2 + 2) + 63) >> 6;
  LODWORD(v13) = ((unsigned __int64)v30 + 63) >> 6;
  if ( v30 <= 0x40 )
  {
LABEL_24:
    if ( !(_DWORD)v13 )
    {
      v21 = v30;
      v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v30;
      goto LABEL_27;
    }
    v14 = &v29;
LABEL_9:
    v15 = 8LL * v8;
    v16 = v8 + 1;
    v17 = 8 * (v16 - v8);
    v18 = &v14[v15 / 0xFFFFFFFFFFFFFFF8LL];
    v19 = 0;
    do
    {
      v20 = 0;
      if ( (unsigned int)v12 > v19 + (unsigned int)v16 )
        v20 = *(_QWORD *)(v15 + *a2 + v17) << (64 - v10);
      ++v19;
      v18[(unsigned __int64)v15 / 8] = (*(_QWORD *)(*a2 + v15) >> v10) | v20;
      v15 += 8;
    }
    while ( v19 < (unsigned int)v13 );
    v21 = v30;
    v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v30;
    if ( v30 > 0x40 )
    {
      v14 = (unsigned __int64 *)v29;
      v23 = ((unsigned __int64)v30 + 63) >> 6;
      goto LABEL_15;
    }
LABEL_27:
    *(_DWORD *)(a1 + 8) = v21;
    v28 = v29 & v22;
LABEL_28:
    *(_QWORD *)a1 = v28;
    return a1;
  }
  v14 = (unsigned __int64 *)v29;
  if ( ((unsigned __int64)v30 + 63) >> 6 )
    goto LABEL_9;
  LODWORD(v23) = 0;
  v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v30;
LABEL_15:
  v14[(unsigned int)(v23 - 1)] &= v22;
  v24 = v30;
  *(_DWORD *)(a1 + 8) = v30;
  if ( v24 <= 0x40 )
  {
    v28 = v29;
    goto LABEL_28;
  }
  sub_16A4FD0(a1, (const void **)&v29);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  return a1;
}

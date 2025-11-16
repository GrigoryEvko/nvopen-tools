// Function: sub_C440A0
// Address: 0xc440a0
//
__int64 __fastcall sub_C440A0(__int64 a1, __int64 *a2, unsigned int a3, unsigned int a4)
{
  unsigned int v7; // edx
  unsigned int v8; // r14d
  unsigned int v9; // eax
  char v10; // r13
  unsigned __int64 v12; // r11
  unsigned __int64 v13; // r9
  unsigned __int64 *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // r10
  __int64 v17; // r15
  unsigned __int64 *v18; // r14
  unsigned int v19; // esi
  __int64 v20; // rdi
  unsigned int v21; // eax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // r8
  __int64 v28; // r8
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
      sub_C43690(a1, v26, 0);
    }
    else
    {
      *(_QWORD *)a1 = v26;
      sub_C43640((unsigned __int64 *)a1);
    }
    return a1;
  }
  v8 = a4 >> 6;
  v9 = (a3 + a4 - 1) >> 6;
  v10 = a4 & 0x3F;
  if ( a4 >> 6 == v9 )
  {
    v27 = *(_QWORD *)(*a2 + 8LL * v8);
    *(_DWORD *)(a1 + 8) = a3;
    v28 = v27 >> v10;
    if ( a3 > 0x40 )
    {
      sub_C43690(a1, v28, 0);
    }
    else
    {
      *(_QWORD *)a1 = v28;
      sub_C43640((unsigned __int64 *)a1);
    }
    return a1;
  }
  if ( (a4 & 0x3F) == 0 )
  {
    sub_C438C0(a1, a3, (_QWORD *)(*a2 + 8LL * v8), v9 - v8 + 1);
    return a1;
  }
  v30 = a3;
  if ( a3 <= 0x40 )
  {
    v29 = 0;
    v12 = (unsigned __int64)(a3 + 63) >> 6;
    v13 = ((unsigned __int64)v7 + 63) >> 6;
    goto LABEL_8;
  }
  sub_C43690((__int64)&v29, 0, 0);
  v13 = ((unsigned __int64)*((unsigned int *)a2 + 2) + 63) >> 6;
  LODWORD(v12) = ((unsigned __int64)v30 + 63) >> 6;
  if ( v30 <= 0x40 )
  {
LABEL_8:
    v14 = &v29;
    if ( !(_DWORD)v12 )
      goto LABEL_13;
    goto LABEL_9;
  }
  v23 = v29;
  v14 = (unsigned __int64 *)v29;
  if ( !(((unsigned __int64)v30 + 63) >> 6) )
  {
    LODWORD(v24) = 0;
    v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v30;
    goto LABEL_16;
  }
LABEL_9:
  v15 = 8LL * v8;
  v16 = v8 + 1;
  v17 = 8 * (v16 - v8);
  v18 = &v14[v15 / 0xFFFFFFFFFFFFFFF8LL];
  v19 = 0;
  do
  {
    v20 = 0;
    if ( (unsigned int)v13 > v19 + (unsigned int)v16 )
      v20 = *(_QWORD *)(v15 + *a2 + v17) << (64 - v10);
    ++v19;
    v18[(unsigned __int64)v15 / 8] = (*(_QWORD *)(*a2 + v15) >> v10) | v20;
    v15 += 8;
  }
  while ( v19 < (unsigned int)v12 );
LABEL_13:
  v21 = v30;
  v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v30;
  if ( v30 )
  {
    if ( v30 > 0x40 )
    {
      v23 = v29;
      v24 = ((unsigned __int64)v30 + 63) >> 6;
LABEL_16:
      *(_QWORD *)(v23 + 8LL * (unsigned int)(v24 - 1)) &= v22;
      v21 = v30;
      goto LABEL_17;
    }
  }
  else
  {
    v22 = 0;
  }
  v29 &= v22;
LABEL_17:
  *(_DWORD *)(a1 + 8) = v21;
  if ( v21 > 0x40 )
  {
    sub_C43780(a1, (const void **)&v29);
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
  }
  else
  {
    *(_QWORD *)a1 = v29;
  }
  return a1;
}

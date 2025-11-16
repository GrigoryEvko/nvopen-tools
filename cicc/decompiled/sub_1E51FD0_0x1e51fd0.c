// Function: sub_1E51FD0
// Address: 0x1e51fd0
//
__int64 __fastcall sub_1E51FD0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdx
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r10d
  __int64 *v8; // r13
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rax
  int v14; // eax
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rsi
  char *v18; // rdi
  __int64 v19; // [rsp+0h] [rbp-A0h] BYREF
  int v20; // [rsp+8h] [rbp-98h]
  char *v21; // [rsp+10h] [rbp-90h]
  __int64 v22; // [rsp+18h] [rbp-88h]
  char v23; // [rsp+20h] [rbp-80h] BYREF
  __int64 v24; // [rsp+40h] [rbp-60h] BYREF
  char *v25; // [rsp+48h] [rbp-58h] BYREF
  __int64 v26; // [rsp+50h] [rbp-50h]
  _BYTE v27[72]; // [rsp+58h] [rbp-48h] BYREF

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  v20 = 0;
  v19 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
LABEL_28:
    sub_177C7D0(a1, 2 * v5);
LABEL_29:
    sub_190E590(a1, &v19, &v24);
    v8 = (__int64 *)v24;
    v4 = v19;
    v15 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
    goto LABEL_15;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v4 == *v10 )
  {
LABEL_3:
    v12 = *((unsigned int *)v10 + 2);
    return *(_QWORD *)(a1 + 32) + 56 * v12 + 8;
  }
  while ( v11 != -8 )
  {
    if ( v11 == -16 && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v4 == *v10 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v15 = (unsigned int)(v14 + 1);
  if ( 4 * (int)v15 >= 3 * v5 )
    goto LABEL_28;
  if ( v5 - *(_DWORD *)(a1 + 20) - (unsigned int)v15 <= v5 >> 3 )
  {
    sub_177C7D0(a1, v5);
    goto LABEL_29;
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  *((_DWORD *)v8 + 2) = v20;
  v16 = *a2;
  v21 = &v23;
  v17 = *(_QWORD *)(a1 + 40);
  v22 = 0x400000000LL;
  v24 = v16;
  v25 = v27;
  v26 = 0x400000000LL;
  if ( v17 == *(_QWORD *)(a1 + 48) )
  {
    sub_1E487D0((__int64 *)(a1 + 32), (char *)v17, (__int64)&v24, v15);
    v18 = v25;
  }
  else
  {
    v18 = v27;
    if ( v17 )
    {
      *(_QWORD *)v17 = v16;
      *(_QWORD *)(v17 + 8) = v17 + 24;
      *(_QWORD *)(v17 + 16) = 0x400000000LL;
      if ( (_DWORD)v26 )
        sub_1E40E30(v17 + 8, &v25, v17 + 24, v15, (int)&v25, v11);
      v17 = *(_QWORD *)(a1 + 40);
      v18 = v25;
    }
    *(_QWORD *)(a1 + 40) = v17 + 56;
  }
  if ( v18 != v27 )
    _libc_free((unsigned __int64)v18);
  v12 = -1227133513 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 3) - 1;
  *((_DWORD *)v8 + 2) = v12;
  return *(_QWORD *)(a1 + 32) + 56 * v12 + 8;
}

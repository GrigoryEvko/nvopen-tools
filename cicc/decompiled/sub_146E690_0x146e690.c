// Function: sub_146E690
// Address: 0x146e690
//
char __fastcall sub_146E690(__int64 a1, __int64 a2)
{
  __int64 *v4; // rbx
  __int64 v5; // rax
  __int64 *i; // r13
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // r8
  unsigned int v11; // edx
  __int64 *v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 *v15; // rax
  int v16; // r10d
  __int64 *v17; // rdi
  int v18; // ecx
  int v19; // ecx
  __int64 v21; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v22[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( !*(_DWORD *)(a2 + 32) )
  {
    v4 = *(__int64 **)(a2 + 40);
    v5 = *(unsigned int *)(a2 + 48);
    for ( i = &v4[v5]; i != v4; LOBYTE(v5) = sub_146E690(a1, v7) )
      v7 = *v4++;
    return v5;
  }
  LOBYTE(v5) = sub_1454560(a1, a2);
  if ( (_BYTE)v5 )
    return v5;
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 32LL))(a2);
  v9 = *(_DWORD *)(a1 + 208);
  v21 = v8;
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 184);
    goto LABEL_24;
  }
  v10 = *(_QWORD *)(a1 + 192);
  v11 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v12 = (__int64 *)(v10 + 56LL * v11);
  v13 = *v12;
  if ( v8 != *v12 )
  {
    v16 = 1;
    v17 = 0;
    while ( v13 != -8 )
    {
      if ( v13 == -16 && !v17 )
        v17 = v12;
      v11 = (v9 - 1) & (v16 + v11);
      v12 = (__int64 *)(v10 + 56LL * v11);
      v13 = *v12;
      if ( v8 == *v12 )
        goto LABEL_8;
      ++v16;
    }
    v18 = *(_DWORD *)(a1 + 200);
    if ( v17 )
      v12 = v17;
    ++*(_QWORD *)(a1 + 184);
    v19 = v18 + 1;
    if ( 4 * v19 < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 204) - v19 > v9 >> 3 )
      {
LABEL_20:
        *(_DWORD *)(a1 + 200) = v19;
        if ( *v12 != -8 )
          --*(_DWORD *)(a1 + 204);
        *v12 = v8;
        v15 = v12 + 3;
        v12[1] = (__int64)(v12 + 3);
        v12[2] = 0x400000000LL;
        goto LABEL_10;
      }
LABEL_25:
      sub_146E440(a1 + 184, v9);
      sub_145F190(a1 + 184, &v21, v22);
      v12 = (__int64 *)v22[0];
      v8 = v21;
      v19 = *(_DWORD *)(a1 + 200) + 1;
      goto LABEL_20;
    }
LABEL_24:
    v9 *= 2;
    goto LABEL_25;
  }
LABEL_8:
  v14 = *((unsigned int *)v12 + 4);
  if ( (unsigned int)v14 >= *((_DWORD *)v12 + 5) )
  {
    sub_16CD150(v12 + 1, v12 + 3, 0, 8);
    v15 = (__int64 *)(v12[1] + 8LL * *((unsigned int *)v12 + 4));
  }
  else
  {
    v15 = (__int64 *)(v12[1] + 8 * v14);
  }
LABEL_10:
  *v15 = a2;
  ++*((_DWORD *)v12 + 4);
  v5 = *(unsigned int *)(a1 + 48);
  if ( (unsigned int)v5 >= *(_DWORD *)(a1 + 52) )
  {
    sub_16CD150(a1 + 40, a1 + 56, 0, 8);
    v5 = *(unsigned int *)(a1 + 48);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v5) = a2;
  ++*(_DWORD *)(a1 + 48);
  return v5;
}

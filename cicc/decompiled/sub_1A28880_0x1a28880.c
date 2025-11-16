// Function: sub_1A28880
// Address: 0x1a28880
//
unsigned __int64 __fastcall sub_1A28880(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 result; // rax
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // r13
  char v11; // al
  int v12; // r8d
  int v13; // r9d
  __int64 *v14; // rbx
  unsigned int v15; // r14d
  unsigned __int64 v16; // rax
  __int64 v17; // rbx
  unsigned int v18; // eax
  int v19; // eax
  unsigned int v20; // esi
  unsigned int v21; // ecx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v25[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( !*(_QWORD *)(a2 + 8) )
    return sub_1A21B40(a1, a2, a3, a4, a5, a6);
  if ( *(_BYTE *)(a2 + 16) == 77 )
  {
    result = sub_15F5600(a2);
    if ( result )
      goto LABEL_4;
  }
  else
  {
    result = sub_1A1ABE0(a2);
    if ( result )
    {
LABEL_4:
      if ( **(_QWORD **)(a1 + 336) == result )
        return sub_386EA80(a1, a2);
      v10 = *(_QWORD *)(a1 + 376);
      result = *(unsigned int *)(v10 + 384);
      if ( (unsigned int)result >= *(_DWORD *)(v10 + 388) )
      {
        sub_16CD150(v10 + 376, (const void *)(v10 + 392), 0, 8, v8, v9);
        result = *(unsigned int *)(v10 + 384);
      }
      *(_QWORD *)(*(_QWORD *)(v10 + 376) + 8 * result) = *(_QWORD *)(a1 + 336);
      ++*(_DWORD *)(v10 + 384);
      return result;
    }
  }
  if ( !*(_BYTE *)(a1 + 344) )
  {
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(a1 + 8) & 3LL | a2 | 4;
    return result;
  }
  v24 = a2;
  v11 = sub_1A26E40(a1 + 464, &v24, v25);
  v14 = (__int64 *)v25[0];
  if ( v11 )
  {
    if ( *(_QWORD *)(v25[0] + 8LL) )
      goto LABEL_13;
    goto LABEL_29;
  }
  v18 = *(_DWORD *)(a1 + 472);
  ++*(_QWORD *)(a1 + 464);
  v19 = (v18 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 472) & 1) != 0 )
  {
    v21 = 12;
    v20 = 4;
  }
  else
  {
    v20 = *(_DWORD *)(a1 + 488);
    v21 = 3 * v20;
  }
  if ( 4 * v19 >= v21 )
  {
    v20 *= 2;
LABEL_35:
    sub_1A28360(a1 + 464, v20);
    sub_1A26E40(a1 + 464, &v24, v25);
    v14 = (__int64 *)v25[0];
    v19 = (*(_DWORD *)(a1 + 472) >> 1) + 1;
    goto LABEL_26;
  }
  if ( v20 - (v19 + *(_DWORD *)(a1 + 476)) <= v20 >> 3 )
    goto LABEL_35;
LABEL_26:
  *(_DWORD *)(a1 + 472) = *(_DWORD *)(a1 + 472) & 1 | (2 * v19);
  if ( *v14 != -8 )
    --*(_DWORD *)(a1 + 476);
  v22 = v24;
  v14[1] = 0;
  *v14 = v22;
LABEL_29:
  v23 = sub_1A20CC0(a2, (unsigned __int64 *)v14 + 1);
  if ( v23 )
  {
    result = *(_QWORD *)(a1 + 8) & 3LL | v23 | 4;
    *(_QWORD *)(a1 + 8) = result;
    return result;
  }
LABEL_13:
  v15 = *(_DWORD *)(a1 + 360);
  if ( v15 <= 0x40 )
  {
    v16 = *(_QWORD *)(a1 + 352);
    goto LABEL_15;
  }
  if ( v15 - (unsigned int)sub_16A57B0(a1 + 352) <= 0x40 )
  {
    v16 = **(_QWORD **)(a1 + 352);
LABEL_15:
    if ( *(_QWORD *)(a1 + 368) > v16 )
      return sub_1A22CF0((_QWORD *)a1, a2, a1 + 352, v14[1], 0, v13);
  }
  v17 = *(_QWORD *)(a1 + 376);
  result = *(unsigned int *)(v17 + 384);
  if ( (unsigned int)result >= *(_DWORD *)(v17 + 388) )
  {
    sub_16CD150(v17 + 376, (const void *)(v17 + 392), 0, 8, v12, v13);
    result = *(unsigned int *)(v17 + 384);
  }
  *(_QWORD *)(*(_QWORD *)(v17 + 376) + 8 * result) = *(_QWORD *)(a1 + 336);
  ++*(_DWORD *)(v17 + 384);
  return result;
}

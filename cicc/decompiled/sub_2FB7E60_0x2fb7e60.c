// Function: sub_2FB7E60
// Address: 0x2fb7e60
//
__int64 __fastcall sub_2FB7E60(__int64 a1, unsigned int a2, int *a3)
{
  __int64 v3; // r14
  __int64 v6; // rcx
  unsigned int v7; // esi
  __int64 v8; // rdi
  int *v9; // r9
  int v10; // r10d
  __int64 v11; // r8
  unsigned int i; // eax
  __int64 v13; // r13
  __int64 v14; // rdx
  int v15; // r11d
  unsigned int v16; // eax
  __int64 *v17; // r13
  unsigned __int64 v18; // r14
  __int64 result; // rax
  __int64 v20; // rax
  __int64 v21; // r8
  unsigned __int64 v22; // r9
  int v23; // eax
  int v24; // edx
  int v25; // eax
  int *v26; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v27; // [rsp+8h] [rbp-28h] BYREF
  int v28; // [rsp+Ch] [rbp-24h]

  v3 = a1 + 392;
  v27 = a2;
  v6 = (unsigned int)*a3;
  v7 = *(_DWORD *)(a1 + 416);
  v28 = *a3;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 392);
    v26 = 0;
    goto LABEL_26;
  }
  v8 = *(_QWORD *)(a1 + 400);
  v9 = 0;
  v10 = 1;
  v11 = v7 - 1;
  for ( i = v11
          & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v6) | ((unsigned __int64)(37 * a2) << 32))) >> 31)
           ^ (756364221 * v6)); ; i = v11 & v16 )
  {
    v13 = 16LL * i;
    v14 = v8 + v13;
    v15 = *(_DWORD *)(v8 + v13);
    if ( a2 == v15 && (_DWORD)v6 == *(_DWORD *)(v14 + 4) )
    {
      v17 = (__int64 *)(v14 + 8);
      goto LABEL_12;
    }
    if ( v15 == -1 )
      break;
    if ( v15 == -2 && *(_DWORD *)(v14 + 4) == -2 && !v9 )
      v9 = (int *)(v8 + v13);
LABEL_9:
    v16 = v10 + i;
    ++v10;
  }
  if ( *(_DWORD *)(v14 + 4) != -1 )
    goto LABEL_9;
  v23 = *(_DWORD *)(a1 + 408);
  if ( !v9 )
    v9 = (int *)(v8 + v13);
  ++*(_QWORD *)(a1 + 392);
  v24 = v23 + 1;
  v26 = v9;
  if ( 4 * (v23 + 1) < 3 * v7 )
  {
    v6 = a2;
    if ( v7 - *(_DWORD *)(a1 + 412) - v24 > v7 >> 3 )
      goto LABEL_20;
    goto LABEL_27;
  }
LABEL_26:
  v7 *= 2;
LABEL_27:
  sub_2FB76D0(v3, v7);
  sub_2FB3720(v3, (int *)&v27, &v26);
  v6 = v27;
  v9 = v26;
  v24 = *(_DWORD *)(a1 + 408) + 1;
LABEL_20:
  *(_DWORD *)(a1 + 408) = v24;
  if ( *v9 != -1 || v9[1] != -1 )
    --*(_DWORD *)(a1 + 412);
  *v9 = v6;
  v25 = v28;
  v17 = (__int64 *)(v9 + 2);
  *((_QWORD *)v9 + 1) = 0;
  v9[1] = v25;
LABEL_12:
  v18 = *v17 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v18 )
  {
    v20 = sub_2DF8570(
            *(_QWORD *)(a1 + 8),
            *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                      + 4LL * (*(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL) + a2)),
            *(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL),
            v6,
            v11,
            (__int64)v9);
    result = sub_2FB21F0((_QWORD *)a1, v20, v18, 0, v21, v22);
    *v17 = 4;
  }
  else
  {
    result = *v17 | 4;
    *v17 = result;
  }
  return result;
}

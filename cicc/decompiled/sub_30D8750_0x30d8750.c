// Function: sub_30D8750
// Address: 0x30d8750
//
_DWORD *__fastcall sub_30D8750(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  int v4; // eax
  unsigned int v5; // esi
  int v6; // r12d
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 *v9; // r10
  int v10; // r14d
  unsigned int v11; // ecx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _DWORD *result; // rax
  int v15; // eax
  int v16; // edx
  __int64 v17; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v18; // [rsp+18h] [rbp-28h] BYREF

  v3 = a1 + 768;
  v17 = a2;
  v4 = sub_DF94A0(*(_QWORD *)(a1 + 8));
  v5 = *(_DWORD *)(a1 + 792);
  v6 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 768);
    v18 = 0;
LABEL_19:
    v5 *= 2;
    goto LABEL_20;
  }
  v7 = v17;
  v8 = *(_QWORD *)(a1 + 776);
  v9 = 0;
  v10 = 1;
  v11 = (v5 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
  v12 = (_QWORD *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( v17 == *v12 )
  {
LABEL_3:
    result = v12 + 1;
    goto LABEL_4;
  }
  while ( v13 != -4096 )
  {
    if ( v13 == -8192 && !v9 )
      v9 = v12;
    v11 = (v5 - 1) & (v10 + v11);
    v12 = (_QWORD *)(v8 + 16LL * v11);
    v13 = *v12;
    if ( v17 == *v12 )
      goto LABEL_3;
    ++v10;
  }
  if ( !v9 )
    v9 = v12;
  v15 = *(_DWORD *)(a1 + 784);
  ++*(_QWORD *)(a1 + 768);
  v16 = v15 + 1;
  v18 = v9;
  if ( 4 * (v15 + 1) >= 3 * v5 )
    goto LABEL_19;
  if ( v5 - *(_DWORD *)(a1 + 788) - v16 <= v5 >> 3 )
  {
LABEL_20:
    sub_2A3BD50(v3, v5);
    sub_30D76A0(v3, &v17, &v18);
    v7 = v17;
    v9 = v18;
    v16 = *(_DWORD *)(a1 + 784) + 1;
  }
  *(_DWORD *)(a1 + 784) = v16;
  if ( *v9 != -4096 )
    --*(_DWORD *)(a1 + 788);
  *v9 = v7;
  result = v9 + 1;
  *((_DWORD *)v9 + 2) = 0;
LABEL_4:
  *result = v6;
  *(_DWORD *)(a1 + 748) += v6;
  return result;
}

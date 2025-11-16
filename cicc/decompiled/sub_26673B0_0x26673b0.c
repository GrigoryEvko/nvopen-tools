// Function: sub_26673B0
// Address: 0x26673b0
//
unsigned __int64 __fastcall sub_26673B0(__int64 a1, unsigned __int64 a2)
{
  bool v3; // r8
  __int64 v4; // rdx
  __int64 v5; // rdi
  unsigned __int64 result; // rax
  unsigned int v7; // ecx
  __int64 v8; // r13
  __int64 v9; // rsi
  int *v10; // r12
  __int64 v11; // rax
  _QWORD *v12; // rsi
  int v13; // r10d
  unsigned __int64 v14; // [rsp+8h] [rbp-48h] BYREF
  __int64 v15; // [rsp+10h] [rbp-40h] BYREF
  __int64 v16; // [rsp+18h] [rbp-38h]
  __int64 v17; // [rsp+20h] [rbp-30h]

  v14 = a2;
  v15 = 0;
  v16 = 0;
  v17 = a2;
  if ( a2 != -8192 && a2 != -4096 && a2 != 0 )
  {
    sub_BD73F0((__int64)&v15);
    result = v17;
    v5 = *(_QWORD *)(a1 + 232);
    v3 = v17 != -8192 && v17 != -4096 && v17 != 0;
    v4 = *(unsigned int *)(a1 + 248);
    if ( !(_DWORD)v4 )
      goto LABEL_25;
  }
  else
  {
    v3 = 0;
    v4 = *(unsigned int *)(a1 + 248);
    v5 = *(_QWORD *)(a1 + 232);
    result = a2;
    if ( !(_DWORD)v4 )
      return result;
  }
  v7 = (v4 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
  v8 = v5 + 32LL * v7;
  v9 = *(_QWORD *)(v8 + 16);
  if ( result == v9 )
  {
LABEL_4:
    if ( !v3 )
      goto LABEL_5;
    goto LABEL_26;
  }
  v13 = 1;
  while ( v9 != -4096 )
  {
    v7 = (v4 - 1) & (v13 + v7);
    v8 = v5 + 32LL * v7;
    v9 = *(_QWORD *)(v8 + 16);
    if ( v9 == result )
      goto LABEL_4;
    ++v13;
  }
LABEL_25:
  v8 = v5 + 32LL * (unsigned int)v4;
  if ( !v3 )
    return result;
LABEL_26:
  result = sub_BD60C0(&v15);
  v5 = *(_QWORD *)(a1 + 232);
  v4 = *(unsigned int *)(a1 + 248);
LABEL_5:
  if ( v8 != v5 + 32 * v4 )
  {
    v10 = sub_220F330(*(int **)(v8 + 24), (_QWORD *)(a1 + 184));
    v11 = *((_QWORD *)v10 + 6);
    if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
      sub_BD60C0((_QWORD *)v10 + 4);
    j_j___libc_free_0((unsigned __int64)v10);
    --*(_QWORD *)(a1 + 216);
    v15 = 0;
    v16 = 0;
    v17 = -8192;
    result = *(_QWORD *)(v8 + 16);
    if ( result != -8192 )
    {
      if ( result != -4096 && result )
        sub_BD60C0((_QWORD *)v8);
      *(_QWORD *)(v8 + 16) = -8192;
      result = v17;
      if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
        result = sub_BD60C0(&v15);
    }
    --*(_DWORD *)(a1 + 240);
    v12 = *(_QWORD **)(a1 + 96);
    ++*(_DWORD *)(a1 + 244);
    if ( v12 == *(_QWORD **)(a1 + 104) )
    {
      return sub_2667110((unsigned __int64 *)(a1 + 88), v12, &v14);
    }
    else
    {
      if ( v12 )
      {
        result = v14;
        v12[1] = 0;
        *v12 = 6;
        v12[2] = result;
        if ( result != 0 && result != -4096 && result != -8192 )
          result = sub_BD73F0((__int64)v12);
        v12 = *(_QWORD **)(a1 + 96);
      }
      *(_QWORD *)(a1 + 96) = v12 + 3;
    }
  }
  return result;
}

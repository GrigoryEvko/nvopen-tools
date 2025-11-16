// Function: sub_14687F0
// Address: 0x14687f0
//
__int64 __fastcall sub_14687F0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  int v10; // edx
  int v11; // eax
  unsigned int v12; // esi
  __int64 v13; // rdi
  int v14; // r10d
  __int64 *v15; // r9
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // rcx
  int v19; // r9d
  int v20; // edi
  int v21; // ecx
  __int64 *v22; // [rsp+8h] [rbp-38h] BYREF
  __int64 v23; // [rsp+10h] [rbp-30h] BYREF
  int v24; // [rsp+18h] [rbp-28h]

  v4 = *(unsigned int *)(a1 + 520);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 504);
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
        return *((unsigned int *)v7 + 2);
    }
    else
    {
      v10 = 1;
      while ( v8 != -8 )
      {
        v19 = v10 + 1;
        v6 = (v4 - 1) & (v10 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v10 = v19;
      }
    }
  }
  v11 = sub_1468AA0(a1, a2);
  v12 = *(_DWORD *)(a1 + 520);
  v23 = a2;
  v24 = v11;
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 496);
    goto LABEL_26;
  }
  v13 = *(_QWORD *)(a1 + 504);
  v14 = 1;
  v15 = 0;
  v16 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v17 = (__int64 *)(v13 + 16LL * v16);
  v18 = *v17;
  if ( a2 != *v17 )
  {
    while ( v18 != -8 )
    {
      if ( v18 == -16 && !v15 )
        v15 = v17;
      v16 = (v12 - 1) & (v14 + v16);
      v17 = (__int64 *)(v13 + 16LL * v16);
      v18 = *v17;
      if ( a2 == *v17 )
        return *((unsigned int *)v17 + 2);
      ++v14;
    }
    v20 = *(_DWORD *)(a1 + 512);
    if ( v15 )
      v17 = v15;
    ++*(_QWORD *)(a1 + 496);
    v21 = v20 + 1;
    if ( 4 * (v20 + 1) < 3 * v12 )
    {
      if ( v12 - *(_DWORD *)(a1 + 516) - v21 > v12 >> 3 )
      {
LABEL_22:
        *(_DWORD *)(a1 + 512) = v21;
        if ( *v17 != -8 )
          --*(_DWORD *)(a1 + 516);
        *v17 = a2;
        *((_DWORD *)v17 + 2) = v24;
        return *((unsigned int *)v17 + 2);
      }
LABEL_27:
      sub_1468630(a1 + 496, v12);
      sub_145FB10(a1 + 496, &v23, &v22);
      v17 = v22;
      a2 = v23;
      v21 = *(_DWORD *)(a1 + 512) + 1;
      goto LABEL_22;
    }
LABEL_26:
    v12 *= 2;
    goto LABEL_27;
  }
  return *((unsigned int *)v17 + 2);
}

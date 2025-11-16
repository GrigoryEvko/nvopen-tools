// Function: sub_1494E70
// Address: 0x1494e70
//
__int64 *__fastcall sub_1494E70(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 v7; // rdx
  __int64 v8; // rdi
  unsigned int v9; // eax
  __int64 *v10; // r12
  __int64 v11; // rcx
  __int64 *result; // rax
  int v13; // edx
  int v14; // r10d
  __int64 *v15; // r9
  int v16; // eax
  int v17; // ecx
  __int64 *v18; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v19[3]; // [rsp+8h] [rbp-18h] BYREF

  v5 = sub_146F1B0(*(_QWORD *)(a1 + 112), a2);
  v6 = *(_DWORD *)(a1 + 24);
  v18 = (__int64 *)v5;
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_18;
  }
  v7 = v5;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v10 = (__int64 *)(v8 + 24LL * v9);
  v11 = *v10;
  if ( v7 != *v10 )
  {
    v14 = 1;
    v15 = 0;
    while ( v11 != -8 )
    {
      if ( !v15 && v11 == -16 )
        v15 = v10;
      v9 = (v6 - 1) & (v14 + v9);
      v10 = (__int64 *)(v8 + 24LL * v9);
      v11 = *v10;
      if ( v7 == *v10 )
        goto LABEL_3;
      ++v14;
    }
    v16 = *(_DWORD *)(a1 + 16);
    if ( v15 )
      v10 = v15;
    ++*(_QWORD *)a1;
    v17 = v16 + 1;
    if ( 4 * (v16 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 20) - v17 > v6 >> 3 )
      {
LABEL_14:
        *(_DWORD *)(a1 + 16) = v17;
        if ( *v10 != -8 )
          --*(_DWORD *)(a1 + 20);
        *v10 = v7;
        result = v18;
        *((_DWORD *)v10 + 2) = 0;
        v10[2] = 0;
        goto LABEL_6;
      }
LABEL_19:
      sub_146EB50(a1, v6);
      sub_145F250(a1, (__int64 *)&v18, v19);
      v10 = (__int64 *)v19[0];
      v7 = (__int64)v18;
      v17 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_14;
    }
LABEL_18:
    v6 *= 2;
    goto LABEL_19;
  }
LABEL_3:
  result = (__int64 *)v10[2];
  if ( result )
  {
    if ( *(_DWORD *)(a1 + 344) == *((_DWORD *)v10 + 2) )
      return result;
    v18 = (__int64 *)v10[2];
  }
  else
  {
    result = (__int64 *)v7;
  }
LABEL_6:
  result = sub_1494E10(*(_QWORD *)(a1 + 112), (__int64)result, *(_QWORD *)(a1 + 120), a1 + 128, a3, a4);
  v13 = *(_DWORD *)(a1 + 344);
  v10[2] = (__int64)result;
  *((_DWORD *)v10 + 2) = v13;
  return result;
}

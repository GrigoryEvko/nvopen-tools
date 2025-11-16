// Function: sub_34FA050
// Address: 0x34fa050
//
__int64 *__fastcall sub_34FA050(__int64 a1, __int64 a2, __int32 a3)
{
  __int64 *result; // rax
  __int64 v5; // rdx
  unsigned __int32 v8; // ecx
  __int64 *v9; // rdi
  int v10; // esi
  __int64 v11; // rcx
  __int64 v12; // r9
  unsigned __int64 v13; // rax
  __int64 i; // rsi
  __int16 v15; // dx
  __int64 v16; // rsi
  __int64 v17; // r8
  unsigned int v18; // ecx
  __int64 *v19; // rdx
  __int64 v20; // r10
  __int64 v21; // r14
  unsigned __int64 v22; // r15
  __int64 *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 *v27; // rsi
  __int64 *v28; // rdx
  int v29; // ecx
  __int64 v30; // rcx
  int v31; // edx
  int v32; // edi
  int v33; // r9d
  int v34; // r11d
  __m128i v35; // [rsp+0h] [rbp-40h] BYREF

  result = (__int64 *)*(unsigned int *)(a1 + 256);
  v5 = *(_QWORD *)(a1 + 240);
  if ( !(_DWORD)result )
    return result;
  v8 = ((_DWORD)result - 1) & (37 * a3);
  v9 = (__int64 *)(v5 + 16LL * v8);
  v10 = *(_DWORD *)v9;
  if ( *(_DWORD *)v9 == a3 )
  {
LABEL_3:
    result = (__int64 *)(v5 + 16LL * (_QWORD)result);
    if ( v9 == result )
      return result;
    v11 = a2;
    v12 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL);
    v13 = a2;
    if ( (*(_DWORD *)(a2 + 44) & 4) != 0 )
    {
      do
        v13 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v13 + 44) & 4) != 0 );
    }
    if ( (*(_DWORD *)(a2 + 44) & 8) != 0 )
    {
      do
        v11 = *(_QWORD *)(v11 + 8);
      while ( (*(_BYTE *)(v11 + 44) & 8) != 0 );
    }
    for ( i = *(_QWORD *)(v11 + 8); i != v13; v13 = *(_QWORD *)(v13 + 8) )
    {
      v15 = *(_WORD *)(v13 + 68);
      if ( (unsigned __int16)(v15 - 14) > 4u && v15 != 24 )
        break;
    }
    v16 = *(unsigned int *)(v12 + 144);
    v17 = *(_QWORD *)(v12 + 128);
    if ( (_DWORD)v16 )
    {
      v18 = (v16 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v19 = (__int64 *)(v17 + 16LL * v18);
      v20 = *v19;
      if ( *v19 == v13 )
      {
LABEL_14:
        v21 = v9[1];
        v22 = v19[1] & 0xFFFFFFFFFFFFFFF8LL;
        v23 = (__int64 *)sub_2E09D00((__int64 *)v21, v22 | 4);
        if ( v23 == (__int64 *)(*(_QWORD *)v21 + 24LL * *(unsigned int *)(v21 + 8))
          || (*(_DWORD *)((*v23 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v23 >> 1) & 3) > (*(_DWORD *)(v22 + 24)
                                                                                                | 2u) )
        {
          v24 = 0;
        }
        else
        {
          v24 = v23[2];
        }
        v35.m128i_i32[0] = a3;
        v35.m128i_i64[1] = v24;
        v25 = sub_34F9630(a1 + 264, &v35);
        v26 = v25;
        if ( *(_BYTE *)(v25 + 28) )
        {
          v27 = *(__int64 **)(v25 + 8);
          result = (__int64 *)*(unsigned int *)(v25 + 20);
          v28 = &v27[(_QWORD)result];
          v29 = (int)result;
          if ( v27 != v28 )
          {
            result = v27;
            while ( a2 != *result )
            {
              if ( v28 == ++result )
                return result;
            }
            v30 = (unsigned int)(v29 - 1);
            *(_DWORD *)(v26 + 20) = v30;
            *result = v27[v30];
            ++*(_QWORD *)v26;
          }
        }
        else
        {
          result = sub_C8CA60(v25, a2);
          if ( result )
          {
            *result = -2;
            ++*(_DWORD *)(v26 + 24);
            ++*(_QWORD *)v26;
          }
        }
        return result;
      }
      v31 = 1;
      while ( v20 != -4096 )
      {
        v34 = v31 + 1;
        v18 = (v16 - 1) & (v31 + v18);
        v19 = (__int64 *)(v17 + 16LL * v18);
        v20 = *v19;
        if ( *v19 == v13 )
          goto LABEL_14;
        v31 = v34;
      }
    }
    v19 = (__int64 *)(v17 + 16 * v16);
    goto LABEL_14;
  }
  v32 = 1;
  while ( v10 != 0x7FFFFFFF )
  {
    v33 = v32 + 1;
    v8 = ((_DWORD)result - 1) & (v32 + v8);
    v9 = (__int64 *)(v5 + 16LL * v8);
    v10 = *(_DWORD *)v9;
    if ( *(_DWORD *)v9 == a3 )
      goto LABEL_3;
    v32 = v33;
  }
  return result;
}

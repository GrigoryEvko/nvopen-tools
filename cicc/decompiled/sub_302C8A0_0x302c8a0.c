// Function: sub_302C8A0
// Address: 0x302c8a0
//
__int64 __fastcall sub_302C8A0(unsigned __int8 *a1, __int64 a2)
{
  __int64 v3; // rbx
  int v4; // eax
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 result; // rax
  __int64 *v8; // rcx
  __int64 v9; // rdx
  unsigned __int8 *v10; // r12
  __int64 v11; // rdi
  int v12; // eax
  int v13; // edx
  __int64 v14; // rdi
  unsigned int v15; // eax
  __int64 *v16; // r9
  __int64 v17; // rcx
  int v18; // edx
  int v19; // r10d
  int v20; // eax
  int v21; // r8d
  __int64 *v22; // rsi
  unsigned __int8 *v23; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v24; // [rsp+8h] [rbp-28h] BYREF

  v3 = (__int64)a1;
  v4 = *a1;
  if ( (_BYTE)v4 == 3 )
  {
    v5 = *(_DWORD *)(a2 + 24);
    v23 = a1;
    if ( v5 )
    {
      v6 = *(_QWORD *)(a2 + 8);
      result = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v8 = (__int64 *)(v6 + 8 * result);
      v9 = *v8;
      if ( v3 == *v8 )
        return result;
      v19 = 1;
      v16 = 0;
      while ( v9 != -4096 )
      {
        if ( v16 || v9 != -8192 )
          v8 = v16;
        result = (v5 - 1) & (v19 + (_DWORD)result);
        v9 = *(_QWORD *)(v6 + 8LL * (unsigned int)result);
        if ( v3 == v9 )
          return result;
        ++v19;
        v16 = v8;
        v8 = (__int64 *)(v6 + 8LL * (unsigned int)result);
      }
      v20 = *(_DWORD *)(a2 + 16);
      if ( !v16 )
        v16 = v8;
      ++*(_QWORD *)a2;
      v18 = v20 + 1;
      v24 = v16;
      if ( 4 * (v20 + 1) < 3 * v5 )
      {
        result = v5 - *(_DWORD *)(a2 + 20) - v18;
        if ( (unsigned int)result <= v5 >> 3 )
        {
          sub_2DD9650(a2, v5);
          sub_3028910(a2, (__int64 *)&v23, &v24);
          result = *(unsigned int *)(a2 + 16);
          v3 = (__int64)v23;
          v16 = v24;
          v18 = result + 1;
        }
LABEL_15:
        *(_DWORD *)(a2 + 16) = v18;
        if ( *v16 != -4096 )
          --*(_DWORD *)(a2 + 20);
        *v16 = v3;
        return result;
      }
    }
    else
    {
      ++*(_QWORD *)a2;
      v24 = 0;
    }
    sub_2DD9650(a2, 2 * v5);
    v12 = *(_DWORD *)(a2 + 24);
    if ( v12 )
    {
      v3 = (__int64)v23;
      v13 = v12 - 1;
      v14 = *(_QWORD *)(a2 + 8);
      v15 = (v12 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v16 = (__int64 *)(v14 + 8LL * v15);
      v17 = *v16;
      if ( (unsigned __int8 *)*v16 == v23 )
      {
LABEL_14:
        result = *(unsigned int *)(a2 + 16);
        v24 = v16;
        v18 = result + 1;
      }
      else
      {
        v21 = 1;
        v22 = 0;
        while ( v17 != -4096 )
        {
          if ( v17 == -8192 && !v22 )
            v22 = v16;
          v15 = v13 & (v21 + v15);
          v16 = (__int64 *)(v14 + 8LL * v15);
          v17 = *v16;
          if ( v23 == (unsigned __int8 *)*v16 )
            goto LABEL_14;
          ++v21;
        }
        result = *(unsigned int *)(a2 + 16);
        if ( !v22 )
          v22 = v16;
        v18 = result + 1;
        v24 = v22;
        v16 = v22;
      }
    }
    else
    {
      result = *(unsigned int *)(a2 + 16);
      v24 = 0;
      v16 = 0;
      v3 = (__int64)v23;
      v18 = result + 1;
    }
    goto LABEL_15;
  }
  result = (unsigned int)(v4 - 22);
  if ( (unsigned __int8)result > 6u )
  {
    result = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
    v10 = &a1[-result];
    if ( (a1[7] & 0x40) != 0 )
    {
      v10 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      v3 = (__int64)&v10[result];
    }
    for ( ; (unsigned __int8 *)v3 != v10; result = sub_302C8A0(v11, a2) )
    {
      v11 = *(_QWORD *)v10;
      v10 += 32;
    }
  }
  return result;
}

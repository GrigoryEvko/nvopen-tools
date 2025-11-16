// Function: sub_3553D90
// Address: 0x3553d90
//
__int64 __fastcall sub_3553D90(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  _QWORD *v5; // rdi
  __int64 *v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 *v11; // r12
  __int64 *v12; // r14
  __int64 v13; // r8
  __int64 *v14; // rdi
  __int64 v15; // rcx
  unsigned int v16; // esi
  __int64 *v17; // r10
  int v18; // edx
  unsigned int v19; // esi
  __int64 v20; // r9
  __int64 *v21; // r11
  int v22; // r13d
  unsigned int v23; // edx
  __int64 *v24; // r8
  __int64 v25; // rdi
  int v26; // eax
  __int64 v27; // r12
  int v28; // r11d
  int v29; // eax
  _QWORD v30[7]; // [rsp+8h] [rbp-38h] BYREF

  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v19 = *(_DWORD *)(a1 + 24);
    if ( v19 )
    {
      v20 = *(_QWORD *)(a1 + 8);
      v21 = 0;
      v22 = 1;
      v23 = (v19 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v24 = (__int64 *)(v20 + 8LL * v23);
      v25 = *v24;
      if ( *a2 == *v24 )
        return result;
      while ( v25 != -4096 )
      {
        if ( v21 || v25 != -8192 )
          v24 = v21;
        v23 = (v19 - 1) & (v22 + v23);
        v25 = *(_QWORD *)(v20 + 8LL * v23);
        if ( *a2 == v25 )
          return result;
        ++v22;
        v21 = v24;
        v24 = (__int64 *)(v20 + 8LL * v23);
      }
      if ( !v21 )
        v21 = v24;
      v26 = result + 1;
      ++*(_QWORD *)a1;
      v30[0] = v21;
      if ( 4 * v26 < 3 * v19 )
      {
        if ( v19 - *(_DWORD *)(a1 + 20) - v26 > v19 >> 3 )
        {
LABEL_24:
          *(_DWORD *)(a1 + 16) = v26;
          if ( *v21 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v21 = *a2;
          result = *(unsigned int *)(a1 + 40);
          v27 = *a2;
          if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
          {
            sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, (__int64)v24, v20);
            result = *(unsigned int *)(a1 + 40);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v27;
          ++*(_DWORD *)(a1 + 40);
          return result;
        }
LABEL_38:
        sub_3553650(a1, v19);
        sub_354AA60(a1, a2, v30);
        v21 = (__int64 *)v30[0];
        v26 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_24;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
      v30[0] = 0;
    }
    v19 *= 2;
    goto LABEL_38;
  }
  v5 = *(_QWORD **)(a1 + 32);
  v7 = &v5[*(unsigned int *)(a1 + 40)];
  result = (__int64)sub_353DA20(v5, (__int64)v7, a2);
  if ( v7 == (__int64 *)result )
  {
    v10 = *a2;
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v8 + 1, 8u, v8, v9);
      v7 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
    }
    *v7 = v10;
    result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = result;
    if ( (unsigned int)result > 8 )
    {
      v11 = *(__int64 **)(a1 + 32);
      v12 = &v11[result];
      while ( 1 )
      {
        v16 = *(_DWORD *)(a1 + 24);
        if ( !v16 )
          break;
        v13 = *(_QWORD *)(a1 + 8);
        result = (v16 - 1) & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
        v14 = (__int64 *)(v13 + 8 * result);
        v15 = *v14;
        if ( *v14 != *v11 )
        {
          v28 = 1;
          v17 = 0;
          while ( v15 != -4096 )
          {
            if ( v15 != -8192 || v17 )
              v14 = v17;
            result = (v16 - 1) & (v28 + (_DWORD)result);
            v15 = *(_QWORD *)(v13 + 8LL * (unsigned int)result);
            if ( *v11 == v15 )
              goto LABEL_9;
            ++v28;
            v17 = v14;
            v14 = (__int64 *)(v13 + 8LL * (unsigned int)result);
          }
          v29 = *(_DWORD *)(a1 + 16);
          if ( !v17 )
            v17 = v14;
          ++*(_QWORD *)a1;
          v18 = v29 + 1;
          v30[0] = v17;
          if ( 4 * (v29 + 1) < 3 * v16 )
          {
            if ( v16 - *(_DWORD *)(a1 + 20) - v18 <= v16 >> 3 )
            {
LABEL_13:
              sub_3553650(a1, v16);
              sub_354AA60(a1, v11, v30);
              v17 = (__int64 *)v30[0];
              v18 = *(_DWORD *)(a1 + 16) + 1;
            }
            *(_DWORD *)(a1 + 16) = v18;
            if ( *v17 != -4096 )
              --*(_DWORD *)(a1 + 20);
            result = *v11;
            *v17 = *v11;
            goto LABEL_9;
          }
LABEL_12:
          v16 *= 2;
          goto LABEL_13;
        }
LABEL_9:
        if ( v12 == ++v11 )
          return result;
      }
      ++*(_QWORD *)a1;
      v30[0] = 0;
      goto LABEL_12;
    }
  }
  return result;
}

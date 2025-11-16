// Function: sub_D2EA30
// Address: 0xd2ea30
//
__int64 __fastcall sub_D2EA30(__int64 a1, __int64 *a2)
{
  _QWORD *v4; // rdi
  __int64 *v6; // rsi
  __int64 result; // rax
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
  __int64 v19; // r9
  unsigned int v20; // esi
  int v21; // eax
  __int64 *v22; // rdx
  int v23; // eax
  __int64 v24; // r8
  __int64 v25; // r12
  int v26; // r11d
  int v27; // eax
  __int64 *v28; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v29[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( *(_DWORD *)(a1 + 16) )
  {
    result = sub_A2AFD0(a1, a2, &v28);
    if ( (_BYTE)result )
      return result;
    v20 = *(_DWORD *)(a1 + 24);
    v21 = *(_DWORD *)(a1 + 16);
    v22 = v28;
    ++*(_QWORD *)a1;
    v23 = v21 + 1;
    v24 = 2 * v20;
    v29[0] = v22;
    if ( 4 * v23 >= 3 * v20 )
    {
      v20 *= 2;
    }
    else if ( v20 - *(_DWORD *)(a1 + 20) - v23 > v20 >> 3 )
    {
LABEL_20:
      *(_DWORD *)(a1 + 16) = v23;
      if ( *v22 != -4096 )
        --*(_DWORD *)(a1 + 20);
      *v22 = *a2;
      result = *(unsigned int *)(a1 + 40);
      v25 = *a2;
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
      {
        sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, v24, v19);
        result = *(unsigned int *)(a1 + 40);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v25;
      ++*(_DWORD *)(a1 + 40);
      return result;
    }
    sub_A35F10(a1, v20);
    sub_A2AFD0(a1, a2, v29);
    v22 = (__int64 *)v29[0];
    v23 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_20;
  }
  v4 = *(_QWORD **)(a1 + 32);
  v6 = &v4[*(unsigned int *)(a1 + 40)];
  result = (__int64)sub_D22DD0(v4, (__int64)v6, a2);
  if ( v6 == (__int64 *)result )
  {
    v10 = *a2;
    if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v8 + 1, 8u, v8, v9);
      v6 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
    }
    *v6 = v10;
    result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = result;
    if ( (unsigned int)result > 4 )
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
        if ( *v11 != *v14 )
        {
          v26 = 1;
          v17 = 0;
          while ( v15 != -4096 )
          {
            if ( v15 != -8192 || v17 )
              v14 = v17;
            result = (v16 - 1) & (v26 + (_DWORD)result);
            v15 = *(_QWORD *)(v13 + 8LL * (unsigned int)result);
            if ( *v11 == v15 )
              goto LABEL_9;
            ++v26;
            v17 = v14;
            v14 = (__int64 *)(v13 + 8LL * (unsigned int)result);
          }
          v27 = *(_DWORD *)(a1 + 16);
          if ( !v17 )
            v17 = v14;
          ++*(_QWORD *)a1;
          v18 = v27 + 1;
          v29[0] = v17;
          if ( 4 * (v27 + 1) < 3 * v16 )
          {
            if ( v16 - *(_DWORD *)(a1 + 20) - v18 <= v16 >> 3 )
            {
LABEL_13:
              sub_A35F10(a1, v16);
              sub_A2AFD0(a1, v11, v29);
              v17 = (__int64 *)v29[0];
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
      v29[0] = 0;
      goto LABEL_12;
    }
  }
  return result;
}

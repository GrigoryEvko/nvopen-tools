// Function: sub_290A430
// Address: 0x290a430
//
__int64 __fastcall sub_290A430(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rcx
  __int64 *v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rdx
  unsigned int v15; // esi
  int v16; // r11d
  __int64 *v17; // r10
  unsigned int v18; // edx
  __int64 *v19; // rdi
  __int64 v20; // rcx
  int v21; // eax
  __int64 v22; // r13
  unsigned __int64 v23; // rdx
  __int64 *v24; // r13
  __int64 *v25; // r14
  __int64 *v26; // rcx
  __int64 v27; // rdi
  unsigned int v28; // esi
  __int64 *v29; // r10
  int v30; // edx
  int v31; // r11d
  int v32; // eax
  const void *v33; // [rsp+18h] [rbp-48h]
  __int64 v34; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v35[7]; // [rsp+28h] [rbp-38h] BYREF

  result = a1 + 48;
  v9 = a2;
  v33 = (const void *)(a1 + 48);
  if ( a2 != a3 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(*(_QWORD *)(v9 + 24) + 40LL);
      result = *(unsigned int *)(a1 + 16);
      v34 = v10;
      if ( (_DWORD)result )
        break;
      v11 = *(unsigned int *)(a1 + 40);
      result = *(_QWORD *)(a1 + 32);
      v12 = (__int64 *)(result + 8 * v11);
      v13 = (8 * v11) >> 3;
      if ( !((8 * v11) >> 5) )
        goto LABEL_28;
      v14 = result + 32 * ((8 * v11) >> 5);
      do
      {
        if ( v10 == *(_QWORD *)result )
          goto LABEL_10;
        if ( v10 == *(_QWORD *)(result + 8) )
        {
          result += 8;
          goto LABEL_10;
        }
        if ( v10 == *(_QWORD *)(result + 16) )
        {
          result += 16;
          goto LABEL_10;
        }
        if ( v10 == *(_QWORD *)(result + 24) )
        {
          result += 24;
          goto LABEL_10;
        }
        result += 32;
      }
      while ( result != v14 );
      v13 = ((__int64)v12 - result) >> 3;
LABEL_28:
      switch ( v13 )
      {
        case 2LL:
          goto LABEL_42;
        case 3LL:
          if ( v10 != *(_QWORD *)result )
          {
            result += 8;
LABEL_42:
            if ( v10 != *(_QWORD *)result )
            {
              result += 8;
LABEL_44:
              if ( v10 != *(_QWORD *)result )
              {
                v23 = v11 + 1;
                if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
                  goto LABEL_46;
                goto LABEL_32;
              }
            }
          }
LABEL_10:
          if ( v12 == (__int64 *)result )
            break;
          goto LABEL_11;
        case 1LL:
          goto LABEL_44;
      }
      v23 = v11 + 1;
      if ( v11 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 44) )
        goto LABEL_32;
LABEL_46:
      sub_C8D5F0(a1 + 32, v33, v23, 8u, a5, a6);
      v12 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
LABEL_32:
      *v12 = v10;
      result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
      *(_DWORD *)(a1 + 40) = result;
      if ( (unsigned int)result > 0x20 )
      {
        v24 = *(__int64 **)(a1 + 32);
        v25 = &v24[result];
        while ( 1 )
        {
          v28 = *(_DWORD *)(a1 + 24);
          if ( !v28 )
            break;
          a6 = v28 - 1;
          a5 = *(_QWORD *)(a1 + 8);
          result = (unsigned int)a6 & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
          v26 = (__int64 *)(a5 + 8 * result);
          v27 = *v26;
          if ( *v24 != *v26 )
          {
            v31 = 1;
            v29 = 0;
            while ( v27 != -4096 )
            {
              if ( v29 || v27 != -8192 )
                v26 = v29;
              result = (unsigned int)a6 & (v31 + (_DWORD)result);
              v27 = *(_QWORD *)(a5 + 8LL * (unsigned int)result);
              if ( *v24 == v27 )
                goto LABEL_35;
              ++v31;
              v29 = v26;
              v26 = (__int64 *)(a5 + 8LL * (unsigned int)result);
            }
            v32 = *(_DWORD *)(a1 + 16);
            if ( !v29 )
              v29 = v26;
            ++*(_QWORD *)a1;
            v30 = v32 + 1;
            v35[0] = v29;
            if ( 4 * (v32 + 1) < 3 * v28 )
            {
              if ( v28 - *(_DWORD *)(a1 + 20) - v30 > v28 >> 3 )
                goto LABEL_56;
              goto LABEL_39;
            }
LABEL_38:
            v28 *= 2;
LABEL_39:
            sub_CF28B0(a1, v28);
            sub_D6B660(a1, v24, v35);
            v29 = (__int64 *)v35[0];
            v30 = *(_DWORD *)(a1 + 16) + 1;
LABEL_56:
            *(_DWORD *)(a1 + 16) = v30;
            if ( *v29 != -4096 )
              --*(_DWORD *)(a1 + 20);
            result = *v24;
            *v29 = *v24;
          }
LABEL_35:
          if ( v25 == ++v24 )
            goto LABEL_11;
        }
        ++*(_QWORD *)a1;
        v35[0] = 0;
        goto LABEL_38;
      }
      do
      {
LABEL_11:
        v9 = *(_QWORD *)(v9 + 8);
        if ( !v9 )
          break;
        result = (unsigned int)**(unsigned __int8 **)(v9 + 24) - 30;
      }
      while ( (unsigned __int8)(**(_BYTE **)(v9 + 24) - 30) > 0xAu );
      if ( a3 == v9 )
        return result;
    }
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      a6 = v15 - 1;
      a5 = *(_QWORD *)(a1 + 8);
      v16 = 1;
      v17 = 0;
      v18 = a6 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v19 = (__int64 *)(a5 + 8LL * v18);
      v20 = *v19;
      if ( v10 == *v19 )
        goto LABEL_11;
      while ( v20 != -4096 )
      {
        if ( v20 != -8192 || v17 )
          v19 = v17;
        v18 = a6 & (v16 + v18);
        v20 = *(_QWORD *)(a5 + 8LL * v18);
        if ( v10 == v20 )
          goto LABEL_11;
        ++v16;
        v17 = v19;
        v19 = (__int64 *)(a5 + 8LL * v18);
      }
      if ( !v17 )
        v17 = v19;
      v21 = result + 1;
      ++*(_QWORD *)a1;
      v35[0] = v17;
      if ( 4 * v21 < 3 * v15 )
      {
        if ( v15 - *(_DWORD *)(a1 + 20) - v21 > v15 >> 3 )
        {
LABEL_22:
          *(_DWORD *)(a1 + 16) = v21;
          if ( *v17 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v17 = v10;
          result = *(unsigned int *)(a1 + 40);
          v22 = v34;
          if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
          {
            sub_C8D5F0(a1 + 32, v33, result + 1, 8u, a5, a6);
            result = *(unsigned int *)(a1 + 40);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v22;
          ++*(_DWORD *)(a1 + 40);
          goto LABEL_11;
        }
LABEL_61:
        sub_CF28B0(a1, v15);
        sub_D6B660(a1, &v34, v35);
        v10 = v34;
        v17 = (__int64 *)v35[0];
        v21 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_22;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
      v35[0] = 0;
    }
    v15 *= 2;
    goto LABEL_61;
  }
  return result;
}

// Function: sub_289F980
// Address: 0x289f980
//
__int64 __fastcall sub_289F980(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 *v5; // r12
  __int64 v7; // r9
  __int64 v8; // r8
  int v9; // r11d
  _QWORD *v10; // r10
  unsigned int v11; // edi
  _QWORD *v12; // rcx
  __int64 v13; // rdx
  unsigned int v14; // esi
  __int64 v15; // r13
  int v16; // edx
  int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // edx
  int v20; // ecx
  __int64 v21; // rdi
  int v22; // r11d
  __int64 v23; // rdx
  int v24; // edx
  int v25; // edx
  __int64 v26; // rdi
  unsigned int v27; // r15d
  _QWORD *v28; // r8
  __int64 v29; // rsi
  const void *v30; // [rsp+8h] [rbp-38h]

  result = a1 + 48;
  v30 = (const void *)(a1 + 48);
  if ( a2 != a3 )
  {
    v5 = a2;
    while ( 1 )
    {
      v14 = *(_DWORD *)(a1 + 24);
      v15 = *v5;
      if ( !v14 )
        break;
      v7 = v14 - 1;
      v8 = *(_QWORD *)(a1 + 8);
      v9 = 1;
      v10 = 0;
      v11 = v7 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v12 = (_QWORD *)(v8 + 8LL * v11);
      v13 = *v12;
      if ( v15 == *v12 )
      {
LABEL_4:
        v5 += 4;
        if ( a3 == v5 )
          return result;
      }
      else
      {
        while ( v13 != -4096 )
        {
          if ( v10 || v13 != -8192 )
            v12 = v10;
          v11 = v7 & (v9 + v11);
          result = v8 + 8LL * v11;
          v13 = *(_QWORD *)result;
          if ( v15 == *(_QWORD *)result )
            goto LABEL_4;
          ++v9;
          v10 = v12;
          v12 = (_QWORD *)(v8 + 8LL * v11);
        }
        result = *(unsigned int *)(a1 + 16);
        if ( !v10 )
          v10 = v12;
        ++*(_QWORD *)a1;
        v20 = result + 1;
        if ( 4 * ((int)result + 1) < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 20) - v20 <= v14 >> 3 )
          {
            sub_CE2A30(a1, v14);
            v24 = *(_DWORD *)(a1 + 24);
            if ( !v24 )
            {
LABEL_46:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v25 = v24 - 1;
            v26 = *(_QWORD *)(a1 + 8);
            result = *(unsigned int *)(a1 + 16);
            v7 = 1;
            v27 = v25 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v28 = 0;
            v10 = (_QWORD *)(v26 + 8LL * v27);
            v20 = result + 1;
            v29 = *v10;
            if ( v15 != *v10 )
            {
              while ( v29 != -4096 )
              {
                if ( v29 == -8192 && !v28 )
                  v28 = v10;
                v27 = v25 & (v7 + v27);
                v10 = (_QWORD *)(v26 + 8LL * v27);
                v29 = *v10;
                if ( v15 == *v10 )
                  goto LABEL_23;
                v7 = (unsigned int)(v7 + 1);
              }
              if ( v28 )
                v10 = v28;
            }
          }
          goto LABEL_23;
        }
LABEL_7:
        sub_CE2A30(a1, 2 * v14);
        v16 = *(_DWORD *)(a1 + 24);
        if ( !v16 )
          goto LABEL_46;
        v17 = v16 - 1;
        v18 = *(_QWORD *)(a1 + 8);
        result = *(unsigned int *)(a1 + 16);
        v19 = (v16 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v10 = (_QWORD *)(v18 + 8LL * v19);
        v20 = result + 1;
        v21 = *v10;
        if ( v15 != *v10 )
        {
          v22 = 1;
          v7 = 0;
          while ( v21 != -4096 )
          {
            if ( !v7 && v21 == -8192 )
              v7 = (__int64)v10;
            v19 = v17 & (v22 + v19);
            v10 = (_QWORD *)(v18 + 8LL * v19);
            v21 = *v10;
            if ( v15 == *v10 )
              goto LABEL_23;
            ++v22;
          }
          if ( v7 )
            v10 = (_QWORD *)v7;
        }
LABEL_23:
        *(_DWORD *)(a1 + 16) = v20;
        if ( *v10 != -4096 )
          --*(_DWORD *)(a1 + 20);
        *v10 = v15;
        v23 = *(unsigned int *)(a1 + 40);
        if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
        {
          result = sub_C8D5F0(a1 + 32, v30, v23 + 1, 8u, v23 + 1, v7);
          v23 = *(unsigned int *)(a1 + 40);
        }
        v5 += 4;
        *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v23) = v15;
        ++*(_DWORD *)(a1 + 40);
        if ( a3 == v5 )
          return result;
      }
    }
    ++*(_QWORD *)a1;
    goto LABEL_7;
  }
  return result;
}

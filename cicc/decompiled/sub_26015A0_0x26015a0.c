// Function: sub_26015A0
// Address: 0x26015a0
//
__int64 __fastcall sub_26015A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r8
  __int64 v6; // rbx
  __int64 v9; // r9
  __int64 v10; // rdi
  int v11; // r11d
  __int64 *v12; // r10
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // r12
  int v17; // eax
  int v18; // ecx
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rsi
  int v22; // edx
  int v23; // r11d
  int v24; // eax
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  unsigned int v28; // r14d
  __int64 *v29; // rdi
  __int64 v30; // rcx
  unsigned int v31; // r11d
  __int64 v32; // [rsp+0h] [rbp-40h]
  __int64 v33; // [rsp+0h] [rbp-40h]
  __int64 v34; // [rsp+0h] [rbp-40h]
  const void *v35; // [rsp+8h] [rbp-38h]

  result = a4 + 16;
  v5 = *(_QWORD *)(a2 + 8);
  v35 = (const void *)(a4 + 16);
  if ( a1 != v5 )
  {
    v6 = a1;
    while ( 1 )
    {
      v15 = *(_DWORD *)(a3 + 24);
      v16 = *(_QWORD *)(*(_QWORD *)(v6 + 16) + 40LL);
      if ( !v15 )
        break;
      v9 = v15 - 1;
      v10 = *(_QWORD *)(a3 + 8);
      v11 = 1;
      v12 = 0;
      v13 = v9 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v14 = (__int64 *)(v10 + 8LL * v13);
      result = *v14;
      if ( v16 == *v14 )
      {
LABEL_4:
        v6 = *(_QWORD *)(v6 + 8);
        if ( v6 == v5 )
          return result;
      }
      else
      {
        while ( result != -4096 )
        {
          if ( v12 || result != -8192 )
            v14 = v12;
          v13 = v9 & (v11 + v13);
          result = *(_QWORD *)(v10 + 8LL * v13);
          if ( v16 == result )
            goto LABEL_4;
          ++v11;
          v12 = v14;
          v14 = (__int64 *)(v10 + 8LL * v13);
        }
        v24 = *(_DWORD *)(a3 + 16);
        if ( !v12 )
          v12 = v14;
        ++*(_QWORD *)a3;
        v22 = v24 + 1;
        if ( 4 * (v24 + 1) < 3 * v15 )
        {
          if ( v15 - *(_DWORD *)(a3 + 20) - v22 <= v15 >> 3 )
          {
            v34 = v5;
            sub_CF28B0(a3, v15);
            v25 = *(_DWORD *)(a3 + 24);
            if ( !v25 )
            {
LABEL_46:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v26 = v25 - 1;
            v27 = *(_QWORD *)(a3 + 8);
            v9 = 1;
            v28 = v26 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
            v5 = v34;
            v12 = (__int64 *)(v27 + 8LL * v28);
            v22 = *(_DWORD *)(a3 + 16) + 1;
            v29 = 0;
            v30 = *v12;
            if ( v16 != *v12 )
            {
              while ( v30 != -4096 )
              {
                if ( v30 == -8192 && !v29 )
                  v29 = v12;
                v31 = v9 + 1;
                v9 = v26 & (v28 + (unsigned int)v9);
                v12 = (__int64 *)(v27 + 8LL * (unsigned int)v9);
                v28 = v9;
                v30 = *v12;
                if ( v16 == *v12 )
                  goto LABEL_23;
                v9 = v31;
              }
              if ( v29 )
                v12 = v29;
            }
          }
          goto LABEL_23;
        }
LABEL_7:
        v32 = v5;
        sub_CF28B0(a3, 2 * v15);
        v17 = *(_DWORD *)(a3 + 24);
        if ( !v17 )
          goto LABEL_46;
        v18 = v17 - 1;
        v19 = *(_QWORD *)(a3 + 8);
        v5 = v32;
        LODWORD(v20) = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v12 = (__int64 *)(v19 + 8LL * (unsigned int)v20);
        v21 = *v12;
        v22 = *(_DWORD *)(a3 + 16) + 1;
        if ( v16 != *v12 )
        {
          v23 = 1;
          v9 = 0;
          while ( v21 != -4096 )
          {
            if ( v21 == -8192 && !v9 )
              v9 = (__int64)v12;
            v20 = v18 & (unsigned int)(v20 + v23);
            v12 = (__int64 *)(v19 + 8 * v20);
            v21 = *v12;
            if ( v16 == *v12 )
              goto LABEL_23;
            ++v23;
          }
          if ( v9 )
            v12 = (__int64 *)v9;
        }
LABEL_23:
        *(_DWORD *)(a3 + 16) = v22;
        if ( *v12 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v12 = v16;
        result = *(unsigned int *)(a4 + 8);
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
          v33 = v5;
          sub_C8D5F0(a4, v35, result + 1, 8u, v5, v9);
          result = *(unsigned int *)(a4 + 8);
          v5 = v33;
        }
        *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = v16;
        ++*(_DWORD *)(a4 + 8);
        v6 = *(_QWORD *)(v6 + 8);
        if ( v6 == v5 )
          return result;
      }
    }
    ++*(_QWORD *)a3;
    goto LABEL_7;
  }
  return result;
}

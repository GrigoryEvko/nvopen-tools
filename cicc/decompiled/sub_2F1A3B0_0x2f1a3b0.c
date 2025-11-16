// Function: sub_2F1A3B0
// Address: 0x2f1a3b0
//
__int64 *__fastcall sub_2F1A3B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 *v4; // rbx
  __int64 v5; // rdx
  __int64 *v6; // r11
  __int64 *result; // rax
  int v8; // r14d
  __int64 v9; // r9
  unsigned int v10; // edi
  __int64 v11; // rcx
  unsigned int v12; // esi
  int v13; // r8d
  __int64 v14; // r13
  int v15; // esi
  int v16; // esi
  __int64 v17; // r9
  unsigned int v18; // ecx
  __int64 *v19; // rdx
  __int64 v20; // rdi
  int v21; // r15d
  __int64 *v22; // r10
  int v23; // eax
  int v24; // ecx
  int v25; // ecx
  __int64 v26; // rdi
  __int64 *v27; // r9
  unsigned int v28; // r15d
  int v29; // r10d
  __int64 v30; // rsi
  __int64 *v31; // [rsp+8h] [rbp-48h]
  __int64 *v32; // [rsp+8h] [rbp-48h]
  int v33; // [rsp+14h] [rbp-3Ch]
  int v34; // [rsp+14h] [rbp-3Ch]
  int v35; // [rsp+14h] [rbp-3Ch]
  __int64 v36; // [rsp+18h] [rbp-38h]

  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v4 = (__int64 *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 120LL))(v3);
  v6 = &v4[v5];
  result = (__int64 *)(a1 + 16);
  v36 = a1 + 16;
  if ( v4 != v6 )
  {
    v8 = 0;
    while ( 1 )
    {
      v12 = *(_DWORD *)(a1 + 40);
      v13 = v8;
      v14 = *v4;
      ++v8;
      if ( !v12 )
        break;
      v9 = *(_QWORD *)(a1 + 24);
      v10 = (v12 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      result = (__int64 *)(v9 + 16LL * v10);
      v11 = *result;
      if ( v14 != *result )
      {
        v34 = 1;
        v19 = 0;
        while ( v11 != -4096 )
        {
          if ( v11 != -8192 || v19 )
            result = v19;
          v10 = (v12 - 1) & (v34 + v10);
          v11 = *(_QWORD *)(v9 + 16LL * v10);
          if ( v14 == v11 )
            goto LABEL_4;
          ++v34;
          v19 = result;
          result = (__int64 *)(v9 + 16LL * v10);
        }
        if ( !v19 )
          v19 = result;
        v23 = *(_DWORD *)(a1 + 32);
        ++*(_QWORD *)(a1 + 16);
        result = (__int64 *)(unsigned int)(v23 + 1);
        if ( 4 * (int)result < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(a1 + 36) - (unsigned int)result <= v12 >> 3 )
          {
            v32 = v6;
            v35 = v13;
            sub_2F1A1D0(v36, v12);
            v24 = *(_DWORD *)(a1 + 40);
            if ( !v24 )
            {
LABEL_45:
              ++*(_DWORD *)(a1 + 32);
              BUG();
            }
            v25 = v24 - 1;
            v26 = *(_QWORD *)(a1 + 24);
            v27 = 0;
            v28 = v25 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v13 = v35;
            v6 = v32;
            v29 = 1;
            result = (__int64 *)(unsigned int)(*(_DWORD *)(a1 + 32) + 1);
            v19 = (__int64 *)(v26 + 16LL * v28);
            v30 = *v19;
            if ( v14 != *v19 )
            {
              while ( v30 != -4096 )
              {
                if ( !v27 && v30 == -8192 )
                  v27 = v19;
                v28 = v25 & (v29 + v28);
                v19 = (__int64 *)(v26 + 16LL * v28);
                v30 = *v19;
                if ( v14 == *v19 )
                  goto LABEL_21;
                ++v29;
              }
              if ( v27 )
                v19 = v27;
            }
          }
          goto LABEL_21;
        }
LABEL_7:
        v31 = v6;
        v33 = v13;
        sub_2F1A1D0(v36, 2 * v12);
        v15 = *(_DWORD *)(a1 + 40);
        if ( !v15 )
          goto LABEL_45;
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 24);
        v13 = v33;
        v6 = v31;
        v18 = v16 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        result = (__int64 *)(unsigned int)(*(_DWORD *)(a1 + 32) + 1);
        v19 = (__int64 *)(v17 + 16LL * v18);
        v20 = *v19;
        if ( v14 != *v19 )
        {
          v21 = 1;
          v22 = 0;
          while ( v20 != -4096 )
          {
            if ( v20 == -8192 && !v22 )
              v22 = v19;
            v18 = v16 & (v21 + v18);
            v19 = (__int64 *)(v17 + 16LL * v18);
            v20 = *v19;
            if ( v14 == *v19 )
              goto LABEL_21;
            ++v21;
          }
          if ( v22 )
            v19 = v22;
        }
LABEL_21:
        *(_DWORD *)(a1 + 32) = (_DWORD)result;
        if ( *v19 != -4096 )
          --*(_DWORD *)(a1 + 36);
        *v19 = v14;
        *((_DWORD *)v19 + 2) = v13;
      }
LABEL_4:
      if ( v6 == ++v4 )
        return result;
    }
    ++*(_QWORD *)(a1 + 16);
    goto LABEL_7;
  }
  return result;
}

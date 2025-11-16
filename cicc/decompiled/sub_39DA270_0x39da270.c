// Function: sub_39DA270
// Address: 0x39da270
//
__int64 *__fastcall sub_39DA270(__int64 a1, __int64 a2)
{
  __int64 (*v3)(); // rax
  __int64 v4; // rax
  int v5; // r14d
  __int64 *v6; // rbx
  __int64 v7; // rdx
  __int64 *v8; // r11
  __int64 *result; // rax
  __int64 v10; // r9
  unsigned int v11; // edi
  __int64 v12; // rcx
  unsigned int v13; // esi
  int v14; // r8d
  __int64 v15; // r13
  int v16; // esi
  int v17; // esi
  __int64 v18; // r9
  unsigned int v19; // ecx
  __int64 *v20; // rdx
  __int64 v21; // rdi
  int v22; // r15d
  __int64 *v23; // r10
  int v24; // eax
  int v25; // ecx
  int v26; // ecx
  __int64 v27; // rdi
  unsigned int v28; // r15d
  int v29; // r9d
  __int64 v30; // rsi
  __int64 *v31; // [rsp+8h] [rbp-48h]
  __int64 *v32; // [rsp+8h] [rbp-48h]
  int v33; // [rsp+14h] [rbp-3Ch]
  int v34; // [rsp+14h] [rbp-3Ch]
  int v35; // [rsp+14h] [rbp-3Ch]
  __int64 v36; // [rsp+18h] [rbp-38h]

  v3 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 112LL);
  if ( v3 == sub_1D00B10 )
    BUG();
  v4 = v3();
  v5 = 0;
  v6 = (__int64 *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 48LL))(v4);
  v8 = &v6[v7];
  result = (__int64 *)(a1 + 8);
  v36 = a1 + 8;
  if ( v6 != v8 )
  {
    while ( 1 )
    {
      v13 = *(_DWORD *)(a1 + 32);
      v14 = v5;
      v15 = *v6;
      ++v5;
      if ( !v13 )
        break;
      v10 = *(_QWORD *)(a1 + 16);
      v11 = (v13 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      result = (__int64 *)(v10 + 16LL * v11);
      v12 = *result;
      if ( v15 != *result )
      {
        v34 = 1;
        v20 = 0;
        while ( v12 != -4 )
        {
          if ( v12 != -8 || v20 )
            result = v20;
          v11 = (v13 - 1) & (v34 + v11);
          v12 = *(_QWORD *)(v10 + 16LL * v11);
          if ( v15 == v12 )
            goto LABEL_5;
          ++v34;
          v20 = result;
          result = (__int64 *)(v10 + 16LL * v11);
        }
        if ( !v20 )
          v20 = result;
        v24 = *(_DWORD *)(a1 + 24);
        ++*(_QWORD *)(a1 + 8);
        result = (__int64 *)(unsigned int)(v24 + 1);
        if ( 4 * (int)result < 3 * v13 )
        {
          if ( v13 - *(_DWORD *)(a1 + 28) - (unsigned int)result <= v13 >> 3 )
          {
            v32 = v8;
            v35 = v14;
            sub_39DA0B0(v36, v13);
            v25 = *(_DWORD *)(a1 + 32);
            if ( !v25 )
            {
LABEL_43:
              ++*(_DWORD *)(a1 + 24);
              BUG();
            }
            v26 = v25 - 1;
            v27 = *(_QWORD *)(a1 + 16);
            v23 = 0;
            v28 = v26 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v14 = v35;
            v8 = v32;
            v29 = 1;
            result = (__int64 *)(unsigned int)(*(_DWORD *)(a1 + 24) + 1);
            v20 = (__int64 *)(v27 + 16LL * v28);
            v30 = *v20;
            if ( v15 != *v20 )
            {
              while ( v30 != -4 )
              {
                if ( !v23 && v30 == -8 )
                  v23 = v20;
                v28 = v26 & (v29 + v28);
                v20 = (__int64 *)(v27 + 16LL * v28);
                v30 = *v20;
                if ( v15 == *v20 )
                  goto LABEL_21;
                ++v29;
              }
LABEL_12:
              if ( v23 )
                v20 = v23;
            }
          }
LABEL_21:
          *(_DWORD *)(a1 + 24) = (_DWORD)result;
          if ( *v20 != -4 )
            --*(_DWORD *)(a1 + 28);
          *v20 = v15;
          *((_DWORD *)v20 + 2) = v14;
          goto LABEL_5;
        }
LABEL_8:
        v31 = v8;
        v33 = v14;
        sub_39DA0B0(v36, 2 * v13);
        v16 = *(_DWORD *)(a1 + 32);
        if ( !v16 )
          goto LABEL_43;
        v17 = v16 - 1;
        v18 = *(_QWORD *)(a1 + 16);
        v14 = v33;
        v8 = v31;
        v19 = v17 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        result = (__int64 *)(unsigned int)(*(_DWORD *)(a1 + 24) + 1);
        v20 = (__int64 *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( v15 != *v20 )
        {
          v22 = 1;
          v23 = 0;
          while ( v21 != -4 )
          {
            if ( v21 == -8 && !v23 )
              v23 = v20;
            v19 = v17 & (v22 + v19);
            v20 = (__int64 *)(v18 + 16LL * v19);
            v21 = *v20;
            if ( v15 == *v20 )
              goto LABEL_21;
            ++v22;
          }
          goto LABEL_12;
        }
        goto LABEL_21;
      }
LABEL_5:
      if ( v8 == ++v6 )
        return result;
    }
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_8;
  }
  return result;
}

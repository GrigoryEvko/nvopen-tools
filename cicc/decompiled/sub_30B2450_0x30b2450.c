// Function: sub_30B2450
// Address: 0x30b2450
//
__int64 __fastcall sub_30B2450(__int64 a1, __int64 a2)
{
  char *v3; // r8
  __int64 *v4; // r9
  __int64 v5; // r10
  __int64 result; // rax
  int v7; // eax
  __int64 *v8; // r12
  __int64 v9; // r11
  __int64 *v10; // r15
  __int64 v11; // r9
  __int64 v12; // r8
  unsigned int v13; // edi
  _QWORD *v14; // rax
  __int64 v15; // rcx
  unsigned int v16; // esi
  __int64 v17; // r13
  int v18; // esi
  int v19; // esi
  __int64 v20; // r8
  unsigned int v21; // ecx
  int v22; // eax
  _QWORD *v23; // rdx
  __int64 v24; // rdi
  int v25; // r14d
  _QWORD *v26; // r10
  int v27; // eax
  int v28; // ecx
  int v29; // ecx
  _QWORD *v30; // r8
  unsigned int v31; // r14d
  __int64 v32; // rdi
  int v33; // r10d
  __int64 v34; // rsi
  __int64 v35; // [rsp+0h] [rbp-40h]
  __int64 v36; // [rsp+0h] [rbp-40h]
  __int64 v37; // [rsp+8h] [rbp-38h]
  int v38; // [rsp+8h] [rbp-38h]
  __int64 v39; // [rsp+8h] [rbp-38h]

  v3 = sub_30B0410(*(char **)(a1 + 96), (char *)(*(_QWORD *)(a1 + 96) + 8LL * *(unsigned int *)(a1 + 104)), a2);
  result = 0;
  if ( v4 == (__int64 *)v3 )
  {
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 108) )
    {
      sub_C8D5F0(a1 + 96, (const void *)(a1 + 112), v5 + 1, 8u, (__int64)v3, (__int64)v4);
      v4 = (__int64 *)(*(_QWORD *)(a1 + 96) + 8LL * *(unsigned int *)(a1 + 104));
    }
    *v4 = a2;
    ++*(_DWORD *)(a1 + 104);
    v7 = *(_DWORD *)(a2 + 56);
    if ( v7 == 3 )
    {
      v8 = *(__int64 **)(a2 + 64);
      v9 = a1 + 192;
      if ( v8 != &v8[*(unsigned int *)(a2 + 72)] )
      {
        v10 = &v8[*(unsigned int *)(a2 + 72)];
        v11 = a2;
        while ( 1 )
        {
          v16 = *(_DWORD *)(a1 + 216);
          v17 = *v8;
          if ( !v16 )
            break;
          v12 = *(_QWORD *)(a1 + 200);
          v13 = (v16 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v14 = (_QWORD *)(v12 + 16LL * v13);
          v15 = *v14;
          if ( v17 != *v14 )
          {
            v38 = 1;
            v23 = 0;
            while ( v15 != -4096 )
            {
              if ( v23 || v15 != -8192 )
                v14 = v23;
              v13 = (v16 - 1) & (v38 + v13);
              v15 = *(_QWORD *)(v12 + 16LL * v13);
              if ( v17 == v15 )
                goto LABEL_8;
              ++v38;
              v23 = v14;
              v14 = (_QWORD *)(v12 + 16LL * v13);
            }
            if ( !v23 )
              v23 = v14;
            v27 = *(_DWORD *)(a1 + 208);
            ++*(_QWORD *)(a1 + 192);
            v22 = v27 + 1;
            if ( 4 * v22 < 3 * v16 )
            {
              if ( v16 - *(_DWORD *)(a1 + 212) - v22 <= v16 >> 3 )
              {
                v36 = v11;
                v39 = v9;
                sub_30B2270(v9, v16);
                v28 = *(_DWORD *)(a1 + 216);
                if ( !v28 )
                {
LABEL_51:
                  ++*(_DWORD *)(a1 + 208);
                  BUG();
                }
                v29 = v28 - 1;
                v30 = 0;
                v9 = v39;
                v31 = v29 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
                v11 = v36;
                v32 = *(_QWORD *)(a1 + 200);
                v33 = 1;
                v22 = *(_DWORD *)(a1 + 208) + 1;
                v23 = (_QWORD *)(v32 + 16LL * v31);
                v34 = *v23;
                if ( v17 != *v23 )
                {
                  while ( v34 != -4096 )
                  {
                    if ( !v30 && v34 == -8192 )
                      v30 = v23;
                    v31 = v29 & (v33 + v31);
                    v23 = (_QWORD *)(v32 + 16LL * v31);
                    v34 = *v23;
                    if ( v17 == *v23 )
                      goto LABEL_27;
                    ++v33;
                  }
                  if ( v30 )
                    v23 = v30;
                }
              }
              goto LABEL_27;
            }
LABEL_11:
            v35 = v11;
            v37 = v9;
            sub_30B2270(v9, 2 * v16);
            v18 = *(_DWORD *)(a1 + 216);
            if ( !v18 )
              goto LABEL_51;
            v19 = v18 - 1;
            v20 = *(_QWORD *)(a1 + 200);
            v9 = v37;
            v11 = v35;
            v21 = v19 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v22 = *(_DWORD *)(a1 + 208) + 1;
            v23 = (_QWORD *)(v20 + 16LL * v21);
            v24 = *v23;
            if ( v17 != *v23 )
            {
              v25 = 1;
              v26 = 0;
              while ( v24 != -4096 )
              {
                if ( !v26 && v24 == -8192 )
                  v26 = v23;
                v21 = v19 & (v25 + v21);
                v23 = (_QWORD *)(v20 + 16LL * v21);
                v24 = *v23;
                if ( v17 == *v23 )
                  goto LABEL_27;
                ++v25;
              }
              if ( v26 )
                v23 = v26;
            }
LABEL_27:
            *(_DWORD *)(a1 + 208) = v22;
            if ( *v23 != -4096 )
              --*(_DWORD *)(a1 + 212);
            *v23 = v17;
            v23[1] = v11;
          }
LABEL_8:
          if ( v10 == ++v8 )
            return 1;
        }
        ++*(_QWORD *)(a1 + 192);
        goto LABEL_11;
      }
      return 1;
    }
    if ( v7 != 4 )
      return 1;
    *(_QWORD *)(a1 + 88) = a2;
    return 1;
  }
  return result;
}

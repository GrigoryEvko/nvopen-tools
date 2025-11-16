// Function: sub_BAF650
// Address: 0xbaf650
//
__int64 __fastcall sub_BAF650(__int64 a1, const void *a2, size_t a3, __int64 a4)
{
  __int64 result; // rax
  _QWORD *v5; // r11
  __int64 *v7; // rbx
  __int64 *v8; // r15
  size_t v9; // r10
  __int64 v10; // r12
  int v11; // r13d
  int v12; // eax
  unsigned int v13; // esi
  __int64 v14; // r8
  _QWORD *v15; // rdx
  int v16; // r11d
  unsigned int v17; // edi
  _QWORD *v18; // rax
  __int64 v19; // rcx
  _QWORD *v20; // rax
  int v21; // eax
  int v22; // eax
  int v23; // r9d
  int v24; // r9d
  __int64 v25; // rdi
  unsigned int v26; // ecx
  __int64 v27; // rsi
  _QWORD *v28; // r11
  int v29; // r9d
  int v30; // r9d
  __int64 v31; // rdi
  unsigned int v32; // ecx
  __int64 v33; // rsi
  __int64 v35; // [rsp+10h] [rbp-60h]
  _QWORD *v37; // [rsp+20h] [rbp-50h]
  unsigned int v38; // [rsp+2Ch] [rbp-44h]
  size_t v39; // [rsp+30h] [rbp-40h]
  size_t v40; // [rsp+30h] [rbp-40h]
  size_t v41; // [rsp+30h] [rbp-40h]
  __int64 v42; // [rsp+38h] [rbp-38h]

  result = a1 + 8;
  v5 = *(_QWORD **)(a1 + 24);
  v35 = a1 + 8;
  if ( v5 == (_QWORD *)(a1 + 8) )
    return result;
  do
  {
    v7 = (__int64 *)v5[7];
    v8 = (__int64 *)v5[8];
    v42 = v5[4];
    if ( v7 == v8 )
      goto LABEL_15;
    v9 = a3;
    v37 = v5;
    v38 = ((0xBF58476D1CE4E5B9LL * v42) >> 31) ^ (484763065 * v42);
    do
    {
      v10 = *v7;
      if ( *v7 )
      {
        v11 = *(_DWORD *)(v10 + 8);
        if ( v11 == 1 && *(_QWORD *)(v10 + 32) == v9 )
        {
          if ( !v9 || (v39 = v9, v12 = memcmp(*(const void **)(v10 + 24), a2, v9), v9 = v39, !v12) )
          {
            v13 = *(_DWORD *)(a4 + 24);
            if ( v13 )
            {
              v14 = *(_QWORD *)(a4 + 8);
              v15 = 0;
              v16 = 1;
              v17 = (v13 - 1) & v38;
              v18 = (_QWORD *)(v14 + 16LL * v17);
              v19 = *v18;
              if ( v42 == *v18 )
              {
LABEL_11:
                v20 = v18 + 1;
LABEL_12:
                *v20 = v10;
                goto LABEL_13;
              }
              while ( v19 != -1 )
              {
                if ( !v15 && v19 == -2 )
                  v15 = v18;
                v17 = (v13 - 1) & (v16 + v17);
                v18 = (_QWORD *)(v14 + 16LL * v17);
                v19 = *v18;
                if ( v42 == *v18 )
                  goto LABEL_11;
                ++v16;
              }
              if ( !v15 )
                v15 = v18;
              v21 = *(_DWORD *)(a4 + 16);
              ++*(_QWORD *)a4;
              v22 = v21 + 1;
              if ( 4 * v22 < 3 * v13 )
              {
                if ( v13 - *(_DWORD *)(a4 + 20) - v22 > v13 >> 3 )
                  goto LABEL_27;
                v41 = v9;
                sub_BAF450(a4, v13);
                v29 = *(_DWORD *)(a4 + 24);
                if ( !v29 )
                {
LABEL_51:
                  ++*(_DWORD *)(a4 + 16);
                  BUG();
                }
                v30 = v29 - 1;
                v31 = *(_QWORD *)(a4 + 8);
                v28 = 0;
                v9 = v41;
                v32 = v30 & v38;
                v22 = *(_DWORD *)(a4 + 16) + 1;
                v15 = (_QWORD *)(v31 + 16LL * (v30 & v38));
                v33 = *v15;
                if ( v42 == *v15 )
                  goto LABEL_27;
                while ( v33 != -1 )
                {
                  if ( v33 == -2 && !v28 )
                    v28 = v15;
                  v32 = v30 & (v11 + v32);
                  v15 = (_QWORD *)(v31 + 16LL * v32);
                  v33 = *v15;
                  if ( v42 == *v15 )
                    goto LABEL_27;
                  ++v11;
                }
                goto LABEL_35;
              }
            }
            else
            {
              ++*(_QWORD *)a4;
            }
            v40 = v9;
            sub_BAF450(a4, 2 * v13);
            v23 = *(_DWORD *)(a4 + 24);
            if ( !v23 )
              goto LABEL_51;
            v24 = v23 - 1;
            v25 = *(_QWORD *)(a4 + 8);
            v9 = v40;
            v26 = v24 & v38;
            v22 = *(_DWORD *)(a4 + 16) + 1;
            v15 = (_QWORD *)(v25 + 16LL * (v24 & v38));
            v27 = *v15;
            if ( v42 == *v15 )
              goto LABEL_27;
            v28 = 0;
            while ( v27 != -1 )
            {
              if ( !v28 && v27 == -2 )
                v28 = v15;
              v26 = v24 & (v11 + v26);
              v15 = (_QWORD *)(v25 + 16LL * v26);
              v27 = *v15;
              if ( v42 == *v15 )
                goto LABEL_27;
              ++v11;
            }
LABEL_35:
            if ( v28 )
              v15 = v28;
LABEL_27:
            *(_DWORD *)(a4 + 16) = v22;
            if ( *v15 != -1 )
              --*(_DWORD *)(a4 + 20);
            v15[1] = 0;
            *v15 = v42;
            v20 = v15 + 1;
            goto LABEL_12;
          }
        }
      }
LABEL_13:
      ++v7;
    }
    while ( v8 != v7 );
    v5 = v37;
LABEL_15:
    result = sub_220EF30(v5);
    v5 = (_QWORD *)result;
  }
  while ( v35 != result );
  return result;
}

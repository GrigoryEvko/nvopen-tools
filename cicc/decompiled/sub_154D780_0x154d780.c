// Function: sub_154D780
// Address: 0x154d780
//
void __fastcall sub_154D780(__int64 a1)
{
  _QWORD *v2; // r14
  _QWORD *v3; // r15
  _QWORD *v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rdx
  unsigned int v7; // esi
  int v8; // r8d
  __int64 v9; // r10
  unsigned int v10; // r9d
  __int64 *v11; // rax
  __int64 v12; // rdi
  int v13; // ecx
  __int64 *v14; // rcx
  int v15; // ecx
  int v16; // edi
  int v17; // esi
  int v18; // esi
  __int64 v19; // r10
  unsigned int v20; // edx
  __int64 v21; // r9
  int v22; // r11d
  __int64 *v23; // rcx
  int v24; // esi
  int v25; // esi
  int v26; // r11d
  __int64 v27; // r10
  unsigned int v28; // edx
  __int64 v29; // r9
  unsigned int v30; // [rsp-50h] [rbp-50h]
  __int64 v31; // [rsp-48h] [rbp-48h]
  int v32; // [rsp-40h] [rbp-40h]
  int v33; // [rsp-3Ch] [rbp-3Ch]

  if ( *(_QWORD *)a1 )
  {
    sub_1648080(a1 + 8, *(_QWORD *)a1, 0);
    v2 = *(_QWORD **)(a1 + 104);
    v3 = *(_QWORD **)(a1 + 112);
    *(_QWORD *)a1 = 0;
    v33 = 0;
    v4 = v2;
    v31 = a1 + 136;
    if ( v2 != v3 )
    {
      while ( 1 )
      {
        v5 = *v4;
        if ( (*(_BYTE *)(*v4 + 9LL) & 4) != 0 )
          goto LABEL_5;
        sub_1643640(*v4);
        if ( v6 )
        {
          *v2++ = v5;
LABEL_5:
          if ( v3 == ++v4 )
            goto LABEL_11;
        }
        else
        {
          v7 = *(_DWORD *)(a1 + 160);
          v8 = v33 + 1;
          if ( !v7 )
          {
            ++*(_QWORD *)(a1 + 136);
            goto LABEL_25;
          }
          v9 = *(_QWORD *)(a1 + 144);
          v10 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( v5 != *v11 )
          {
            v32 = 1;
            v14 = 0;
            while ( v12 != -8 )
            {
              if ( !v14 && v12 == -16 )
                v14 = v11;
              v10 = (v7 - 1) & (v32 + v10);
              v11 = (__int64 *)(v9 + 16LL * v10);
              v12 = *v11;
              if ( v5 == *v11 )
                goto LABEL_10;
              ++v32;
            }
            if ( v14 )
              v11 = v14;
            v15 = *(_DWORD *)(a1 + 152);
            ++*(_QWORD *)(a1 + 136);
            v16 = v15 + 1;
            if ( 4 * (v15 + 1) < 3 * v7 )
            {
              if ( v7 - *(_DWORD *)(a1 + 156) - v16 <= v7 >> 3 )
              {
                v30 = ((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4);
                sub_154D5C0(v31, v7);
                v24 = *(_DWORD *)(a1 + 160);
                if ( !v24 )
                {
LABEL_50:
                  ++*(_DWORD *)(a1 + 152);
                  BUG();
                }
                v25 = v24 - 1;
                v8 = v33 + 1;
                v26 = 1;
                v27 = *(_QWORD *)(a1 + 144);
                v28 = v25 & v30;
                v16 = *(_DWORD *)(a1 + 152) + 1;
                v23 = 0;
                v11 = (__int64 *)(v27 + 16LL * (v25 & v30));
                v29 = *v11;
                if ( v5 != *v11 )
                {
                  while ( v29 != -8 )
                  {
                    if ( !v23 && v29 == -16 )
                      v23 = v11;
                    v28 = v25 & (v26 + v28);
                    v11 = (__int64 *)(v27 + 16LL * v28);
                    v29 = *v11;
                    if ( v5 == *v11 )
                      goto LABEL_21;
                    ++v26;
                  }
                  goto LABEL_29;
                }
              }
              goto LABEL_21;
            }
LABEL_25:
            sub_154D5C0(v31, 2 * v7);
            v17 = *(_DWORD *)(a1 + 160);
            if ( !v17 )
              goto LABEL_50;
            v18 = v17 - 1;
            v19 = *(_QWORD *)(a1 + 144);
            v8 = v33 + 1;
            v20 = v18 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
            v16 = *(_DWORD *)(a1 + 152) + 1;
            v11 = (__int64 *)(v19 + 16LL * v20);
            v21 = *v11;
            if ( v5 != *v11 )
            {
              v22 = 1;
              v23 = 0;
              while ( v21 != -8 )
              {
                if ( v21 == -16 && !v23 )
                  v23 = v11;
                v20 = v18 & (v22 + v20);
                v11 = (__int64 *)(v19 + 16LL * v20);
                v21 = *v11;
                if ( v5 == *v11 )
                  goto LABEL_21;
                ++v22;
              }
LABEL_29:
              if ( v23 )
                v11 = v23;
            }
LABEL_21:
            *(_DWORD *)(a1 + 152) = v16;
            if ( *v11 != -8 )
              --*(_DWORD *)(a1 + 156);
            *v11 = v5;
            *((_DWORD *)v11 + 2) = 0;
          }
LABEL_10:
          v13 = v33;
          ++v4;
          v33 = v8;
          *((_DWORD *)v11 + 2) = v13;
          if ( v3 == v4 )
          {
LABEL_11:
            if ( v2 != *(_QWORD **)(a1 + 112) )
              *(_QWORD *)(a1 + 112) = v2;
            return;
          }
        }
      }
    }
  }
}

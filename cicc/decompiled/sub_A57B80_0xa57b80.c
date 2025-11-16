// Function: sub_A57B80
// Address: 0xa57b80
//
void __fastcall sub_A57B80(__int64 a1)
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
  _QWORD *v11; // rax
  __int64 v12; // rdi
  _DWORD *v13; // rax
  int v14; // ecx
  _QWORD *v15; // rcx
  int v16; // eax
  int v17; // eax
  int v18; // esi
  int v19; // esi
  __int64 v20; // r9
  unsigned int v21; // edx
  __int64 v22; // rdi
  int v23; // r10d
  _QWORD *v24; // r11
  int v25; // esi
  int v26; // esi
  __int64 v27; // r9
  int v28; // r10d
  unsigned int v29; // edx
  __int64 v30; // rdi
  unsigned int v31; // [rsp-50h] [rbp-50h]
  __int64 v32; // [rsp-48h] [rbp-48h]
  int v33; // [rsp-40h] [rbp-40h]
  int v34; // [rsp-3Ch] [rbp-3Ch]

  if ( *(_QWORD *)a1 )
  {
    sub_BD22F0(a1 + 8, *(_QWORD *)a1, 0);
    v2 = *(_QWORD **)(a1 + 136);
    v3 = *(_QWORD **)(a1 + 144);
    *(_QWORD *)a1 = 0;
    v34 = 0;
    v4 = v2;
    v32 = a1 + 168;
    if ( v2 != v3 )
    {
      while ( 1 )
      {
        v5 = *v4;
        if ( (*(_BYTE *)(*v4 + 9LL) & 4) != 0 )
          goto LABEL_5;
        sub_BCB490(*v4);
        if ( v6 )
        {
          *v2++ = v5;
LABEL_5:
          if ( v3 == ++v4 )
            goto LABEL_12;
        }
        else
        {
          v7 = *(_DWORD *)(a1 + 192);
          v8 = v34 + 1;
          if ( !v7 )
          {
            ++*(_QWORD *)(a1 + 168);
            goto LABEL_26;
          }
          v9 = *(_QWORD *)(a1 + 176);
          v10 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
          v11 = (_QWORD *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( v5 != *v11 )
          {
            v33 = 1;
            v15 = 0;
            while ( v12 != -4096 )
            {
              if ( !v15 && v12 == -8192 )
                v15 = v11;
              v10 = (v7 - 1) & (v33 + v10);
              v11 = (_QWORD *)(v9 + 16LL * v10);
              v12 = *v11;
              if ( v5 == *v11 )
                goto LABEL_10;
              ++v33;
            }
            if ( !v15 )
              v15 = v11;
            v16 = *(_DWORD *)(a1 + 184);
            ++*(_QWORD *)(a1 + 168);
            v17 = v16 + 1;
            if ( 4 * v17 < 3 * v7 )
            {
              if ( v7 - *(_DWORD *)(a1 + 188) - v17 <= v7 >> 3 )
              {
                v31 = ((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4);
                sub_A579A0(v32, v7);
                v25 = *(_DWORD *)(a1 + 192);
                if ( !v25 )
                {
LABEL_51:
                  ++*(_DWORD *)(a1 + 184);
                  BUG();
                }
                v26 = v25 - 1;
                v27 = *(_QWORD *)(a1 + 176);
                v24 = 0;
                v8 = v34 + 1;
                v28 = 1;
                v29 = v26 & v31;
                v17 = *(_DWORD *)(a1 + 184) + 1;
                v15 = (_QWORD *)(v27 + 16LL * (v26 & v31));
                v30 = *v15;
                if ( v5 != *v15 )
                {
                  while ( v30 != -4096 )
                  {
                    if ( !v24 && v30 == -8192 )
                      v24 = v15;
                    v29 = v26 & (v28 + v29);
                    v15 = (_QWORD *)(v27 + 16LL * v29);
                    v30 = *v15;
                    if ( v5 == *v15 )
                      goto LABEL_22;
                    ++v28;
                  }
                  goto LABEL_30;
                }
              }
              goto LABEL_22;
            }
LABEL_26:
            sub_A579A0(v32, 2 * v7);
            v18 = *(_DWORD *)(a1 + 192);
            if ( !v18 )
              goto LABEL_51;
            v19 = v18 - 1;
            v20 = *(_QWORD *)(a1 + 176);
            v8 = v34 + 1;
            v21 = v19 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
            v17 = *(_DWORD *)(a1 + 184) + 1;
            v15 = (_QWORD *)(v20 + 16LL * v21);
            v22 = *v15;
            if ( v5 != *v15 )
            {
              v23 = 1;
              v24 = 0;
              while ( v22 != -4096 )
              {
                if ( v22 == -8192 && !v24 )
                  v24 = v15;
                v21 = v19 & (v23 + v21);
                v15 = (_QWORD *)(v20 + 16LL * v21);
                v22 = *v15;
                if ( v5 == *v15 )
                  goto LABEL_22;
                ++v23;
              }
LABEL_30:
              if ( v24 )
                v15 = v24;
            }
LABEL_22:
            *(_DWORD *)(a1 + 184) = v17;
            if ( *v15 != -4096 )
              --*(_DWORD *)(a1 + 188);
            *v15 = v5;
            v13 = v15 + 1;
            *((_DWORD *)v15 + 2) = 0;
            goto LABEL_11;
          }
LABEL_10:
          v13 = v11 + 1;
LABEL_11:
          v14 = v34;
          ++v4;
          v34 = v8;
          *v13 = v14;
          if ( v3 == v4 )
          {
LABEL_12:
            if ( v2 != *(_QWORD **)(a1 + 144) )
              *(_QWORD *)(a1 + 144) = v2;
            return;
          }
        }
      }
    }
  }
}

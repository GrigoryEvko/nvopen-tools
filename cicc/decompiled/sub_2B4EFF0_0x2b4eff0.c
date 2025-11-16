// Function: sub_2B4EFF0
// Address: 0x2b4eff0
//
__int64 __fastcall sub_2B4EFF0(__int64 ***a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rbx
  __int64 *v5; // r13
  int v6; // r12d
  __int64 v7; // r10
  unsigned int v8; // esi
  _QWORD *v9; // rcx
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // rax
  unsigned int v13; // r8d
  int v14; // esi
  int v15; // esi
  __int64 v16; // r10
  unsigned int v17; // ecx
  int v18; // eax
  _QWORD *v19; // rdi
  __int64 v20; // r9
  int v21; // r15d
  _QWORD *v22; // r11
  __int64 v23; // r8
  __int64 v24; // rsi
  int v25; // edi
  __int64 v26; // r9
  int v27; // edi
  unsigned int v28; // esi
  __int64 *v29; // rcx
  __int64 v30; // r11
  __int64 v31; // r10
  __int64 v32; // rsi
  unsigned int v33; // r8d
  __int64 *v34; // rcx
  __int64 v35; // r15
  int v37; // r15d
  int v38; // eax
  __int64 v39; // rax
  int v40; // esi
  int v41; // esi
  __int64 v42; // r10
  int v43; // r15d
  unsigned int v44; // ecx
  __int64 v45; // r9
  int v46; // ecx
  int v47; // r10d
  int v48; // ecx
  int v49; // r11d
  __int64 v50; // [rsp+8h] [rbp-38h]
  __int64 v51; // [rsp+8h] [rbp-38h]

  v4 = **a1;
  v5 = &v4[4 * *((unsigned int *)*a1 + 2)];
  if ( v4 != v5 )
  {
    v6 = 0;
    do
    {
      while ( 1 )
      {
        v11 = v4[1];
        v12 = *v4;
        if ( !v11 )
          goto LABEL_7;
        if ( *(_BYTE *)v11 != 84 )
          goto LABEL_7;
        v23 = *(_QWORD *)(v11 + 40);
        v24 = *(_QWORD *)(*(_QWORD *)a2 + 3312LL);
        v25 = *(_DWORD *)(v24 + 24);
        v26 = *(_QWORD *)(v24 + 8);
        if ( !v25 )
          goto LABEL_7;
        v27 = v25 - 1;
        v28 = v27 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v29 = (__int64 *)(v26 + 16LL * v28);
        v30 = *v29;
        if ( v23 == *v29 )
          break;
        v46 = 1;
        while ( v30 != -4096 )
        {
          v47 = v46 + 1;
          v28 = v27 & (v46 + v28);
          v29 = (__int64 *)(v26 + 16LL * v28);
          v30 = *v29;
          if ( v23 == *v29 )
            goto LABEL_18;
          v46 = v47;
        }
LABEL_7:
        v13 = *(_DWORD *)(a3 + 24);
        if ( !v13 )
        {
          ++*(_QWORD *)a3;
          goto LABEL_9;
        }
        v7 = *(_QWORD *)(a3 + 8);
        v8 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v9 = (_QWORD *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( v12 != *v9 )
        {
          v37 = 1;
          v19 = 0;
          while ( v10 != -4096 )
          {
            if ( v19 || v10 != -8192 )
              v9 = v19;
            v8 = (v13 - 1) & (v37 + v8);
            v10 = *(_QWORD *)(v7 + 16LL * v8);
            if ( v12 == v10 )
              goto LABEL_4;
            ++v37;
            v19 = v9;
            v9 = (_QWORD *)(v7 + 16LL * v8);
          }
          v38 = *(_DWORD *)(a3 + 16);
          if ( !v19 )
            v19 = v9;
          ++*(_QWORD *)a3;
          v18 = v38 + 1;
          if ( 4 * v18 >= 3 * v13 )
          {
LABEL_9:
            v50 = a3;
            sub_D39D40(a3, 2 * v13);
            a3 = v50;
            v14 = *(_DWORD *)(v50 + 24);
            if ( !v14 )
              goto LABEL_59;
            v15 = v14 - 1;
            v16 = *(_QWORD *)(v50 + 8);
            v17 = v15 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
            v18 = *(_DWORD *)(v50 + 16) + 1;
            v19 = (_QWORD *)(v16 + 16LL * v17);
            v20 = *v19;
            if ( *v19 != *v4 )
            {
              v21 = 1;
              v22 = 0;
              while ( v20 != -4096 )
              {
                if ( !v22 && v20 == -8192 )
                  v22 = v19;
                v17 = v15 & (v21 + v17);
                v19 = (_QWORD *)(v16 + 16LL * v17);
                v20 = *v19;
                if ( *v4 == *v19 )
                  goto LABEL_30;
                ++v21;
              }
              goto LABEL_13;
            }
          }
          else if ( v13 - *(_DWORD *)(a3 + 20) - v18 <= v13 >> 3 )
          {
            v51 = a3;
            sub_D39D40(a3, v13);
            a3 = v51;
            v40 = *(_DWORD *)(v51 + 24);
            if ( !v40 )
            {
LABEL_59:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v41 = v40 - 1;
            v42 = *(_QWORD *)(v51 + 8);
            v43 = 1;
            v22 = 0;
            v44 = v41 & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
            v18 = *(_DWORD *)(v51 + 16) + 1;
            v19 = (_QWORD *)(v42 + 16LL * v44);
            v45 = *v19;
            if ( *v19 != *v4 )
            {
              while ( v45 != -4096 )
              {
                if ( !v22 && v45 == -8192 )
                  v22 = v19;
                v44 = v41 & (v43 + v44);
                v19 = (_QWORD *)(v42 + 16LL * v44);
                v45 = *v19;
                if ( *v4 == *v19 )
                  goto LABEL_30;
                ++v43;
              }
LABEL_13:
              if ( v22 )
                v19 = v22;
            }
          }
LABEL_30:
          *(_DWORD *)(a3 + 16) = v18;
          if ( *v19 != -4096 )
            --*(_DWORD *)(a3 + 20);
          v39 = *v4;
          *((_DWORD *)v19 + 2) = v6;
          *v19 = v39;
        }
LABEL_4:
        v4 += 4;
        ++v6;
        if ( v5 == v4 )
          return a2;
      }
LABEL_18:
      v31 = v29[1];
      if ( !v31 )
        goto LABEL_7;
      v32 = *(_QWORD *)(v12 + 40);
      if ( v23 == v32 )
        goto LABEL_4;
      v33 = v27 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
      v34 = (__int64 *)(v26 + 16LL * v33);
      v35 = *v34;
      if ( v32 != *v34 )
      {
        v48 = 1;
        while ( v35 != -4096 )
        {
          v49 = v48 + 1;
          v33 = v27 & (v48 + v33);
          v34 = (__int64 *)(v26 + 16LL * v33);
          v35 = *v34;
          if ( v32 == *v34 )
            goto LABEL_21;
          v48 = v49;
        }
        goto LABEL_7;
      }
LABEL_21:
      if ( v31 != v34[1] )
        goto LABEL_7;
      v4 += 4;
      ++v6;
    }
    while ( v5 != v4 );
  }
  return a2;
}

// Function: sub_CFD0B0
// Address: 0xcfd0b0
//
_QWORD *__fastcall sub_CFD0B0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 v5; // r15
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  _QWORD *i; // rcx
  char v11; // dl
  __int64 v12; // rcx
  __int64 j; // r15
  __int64 v14; // rdx
  int v15; // esi
  int v16; // esi
  __int64 v17; // r9
  unsigned int v18; // edi
  __int64 v19; // rcx
  __int64 v20; // r14
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rdx
  unsigned int v25; // r13d
  _QWORD *v26; // r13
  __int64 v27; // rdi
  _QWORD *v28; // r14
  __int64 v29; // rdx
  __int64 v30; // r8
  __int64 v31; // r10
  __int64 k; // r10
  __int64 v33; // rsi
  _QWORD *v34; // r14
  __int64 v35; // rdx
  _QWORD *v36; // r13
  __int64 v37; // rcx
  __int64 v38; // rdx
  _QWORD *m; // rcx
  char v40; // dl
  int v41; // r11d
  __int64 v42; // r10
  __int64 v43; // rsi
  __int64 v44; // [rsp+8h] [rbp-B8h]
  __int64 v45; // [rsp+10h] [rbp-B0h]
  __int64 v46; // [rsp+18h] [rbp-A8h]
  __int64 v47; // [rsp+20h] [rbp-A0h]
  __int64 v48; // [rsp+28h] [rbp-98h]
  _QWORD v49[2]; // [rsp+38h] [rbp-88h] BYREF
  __int64 v50; // [rsp+48h] [rbp-78h]
  __int64 v51; // [rsp+50h] [rbp-70h]
  void *v52; // [rsp+60h] [rbp-60h]
  _QWORD v53[2]; // [rsp+68h] [rbp-58h] BYREF
  __int64 v54; // [rsp+78h] [rbp-48h]
  __int64 v55; // [rsp+80h] [rbp-40h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v48 = v5;
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(88LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v8 = *(unsigned int *)(a1 + 24);
    v53[0] = 2;
    v55 = 0;
    v47 = 88 * v4;
    v9 = 88 * v4 + v5;
    for ( i = &result[11 * v8]; i != result; result += 11 )
    {
      if ( result )
      {
        v11 = v53[0];
        result[2] = 0;
        result[3] = -4096;
        *result = &unk_49DDAE8;
        result[1] = v11 & 6;
        result[4] = v55;
      }
    }
    v49[1] = 0;
    v49[0] = 2;
    v50 = -4096;
    v51 = 0;
    v53[0] = 2;
    v53[1] = 0;
    v54 = -8192;
    v52 = &unk_49DDAE8;
    v55 = 0;
    if ( v9 != v5 )
    {
      v12 = -4096;
      for ( j = v5 + 56; ; j += 88 )
      {
        v14 = *(_QWORD *)(j - 32);
        if ( v14 != v12 )
        {
          v12 = v54;
          if ( v14 != v54 )
          {
            v15 = *(_DWORD *)(a1 + 24);
            if ( !v15 )
              BUG();
            v16 = v15 - 1;
            v17 = *(_QWORD *)(a1 + 8);
            v18 = v16 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v19 = 11LL * v18;
            v20 = v17 + 88LL * v18;
            v21 = *(_QWORD *)(v20 + 24);
            if ( v14 != v21 )
            {
              v41 = 1;
              v42 = 0;
              while ( v21 != -4096 )
              {
                if ( !v42 && v21 == -8192 )
                  v42 = v20;
                v18 = v16 & (v41 + v18);
                v19 = 11LL * v18;
                v20 = v17 + 88LL * v18;
                v21 = *(_QWORD *)(v20 + 24);
                if ( v14 == v21 )
                  goto LABEL_14;
                ++v41;
              }
              if ( v42 )
              {
                v43 = *(_QWORD *)(v42 + 24);
                v20 = v42;
              }
              else
              {
                v43 = *(_QWORD *)(v20 + 24);
              }
              if ( v14 != v43 )
              {
                LOBYTE(v17) = v43 != 0;
                if ( v43 != -4096 && v43 != 0 && v43 != -8192 )
                {
                  sub_BD60C0((_QWORD *)(v20 + 8));
                  v14 = *(_QWORD *)(j - 32);
                }
                *(_QWORD *)(v20 + 24) = v14;
                if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
                  sub_BD6050((unsigned __int64 *)(v20 + 8), *(_QWORD *)(j - 48) & 0xFFFFFFFFFFFFFFF8LL);
              }
            }
LABEL_14:
            v22 = *(_QWORD *)(j - 24);
            v23 = j - 16;
            *(_QWORD *)(v20 + 48) = 0x100000000LL;
            *(_QWORD *)(v20 + 32) = v22;
            v24 = v20 + 56;
            *(_QWORD *)(v20 + 40) = v20 + 56;
            v25 = *(_DWORD *)(j - 8);
            if ( v20 + 40 != j - 16 && v25 )
            {
              v23 = *(_QWORD *)(j - 16);
              if ( v23 == j )
              {
                v23 = v25;
                v30 = j;
                v31 = 1;
                if ( v25 != 1 )
                {
                  sub_CFC2E0(v20 + 40, v25, v24, v19, j, v17);
                  v24 = *(_QWORD *)(v20 + 40);
                  v30 = *(_QWORD *)(j - 16);
                  v31 = *(unsigned int *)(j - 8);
                }
                for ( k = v30 + 32 * v31; k != v30; v24 += 32 )
                {
                  if ( v24 )
                  {
                    *(_QWORD *)v24 = 4;
                    *(_QWORD *)(v24 + 8) = 0;
                    v33 = *(_QWORD *)(v30 + 16);
                    *(_QWORD *)(v24 + 16) = v33;
                    if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
                    {
                      v44 = k;
                      v45 = v30;
                      v46 = v24;
                      sub_BD6050((unsigned __int64 *)v24, *(_QWORD *)v30 & 0xFFFFFFFFFFFFFFF8LL);
                      k = v44;
                      v30 = v45;
                      v24 = v46;
                    }
                    v23 = *(unsigned int *)(v30 + 24);
                    *(_DWORD *)(v24 + 24) = v23;
                  }
                  v30 += 32;
                }
                *(_DWORD *)(v20 + 48) = v25;
                v34 = *(_QWORD **)(j - 16);
                v35 = 4LL * *(unsigned int *)(j - 8);
                v36 = &v34[v35];
                while ( v34 != v36 )
                {
                  v37 = *(v36 - 2);
                  v36 -= 4;
                  LOBYTE(v23) = v37 != -4096;
                  if ( ((unsigned __int8)v23 & (v37 != 0)) != 0 && v37 != -8192 )
                    sub_BD60C0(v36);
                }
                *(_DWORD *)(j - 8) = 0;
              }
              else
              {
                *(_QWORD *)(v20 + 40) = v23;
                *(_DWORD *)(v20 + 48) = *(_DWORD *)(j - 8);
                *(_DWORD *)(v20 + 52) = *(_DWORD *)(j - 4);
                *(_QWORD *)(j - 16) = j;
                *(_DWORD *)(j - 4) = 0;
                *(_DWORD *)(j - 8) = 0;
              }
            }
            ++*(_DWORD *)(a1 + 16);
            v26 = *(_QWORD **)(j - 16);
            v27 = 4LL * *(unsigned int *)(j - 8);
            v28 = &v26[v27];
            if ( v26 != &v26[v27] )
            {
              do
              {
                v29 = *(v28 - 2);
                v28 -= 4;
                LOBYTE(v23) = v29 != 0;
                if ( v29 != -4096 && v29 != 0 && v29 != -8192 )
                  sub_BD60C0(v28);
              }
              while ( v26 != v28 );
              v28 = *(_QWORD **)(j - 16);
            }
            if ( v28 != (_QWORD *)j )
              _libc_free(v28, v23);
            v12 = *(_QWORD *)(j - 32);
          }
        }
        *(_QWORD *)(j - 56) = &unk_49DB368;
        if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
          sub_BD60C0((_QWORD *)(j - 48));
        if ( v9 == j + 32 )
          break;
        v12 = v50;
      }
      v52 = &unk_49DB368;
      if ( v54 != 0 && v54 != -4096 && v54 != -8192 )
        sub_BD60C0(v53);
    }
    if ( v50 != 0 && v50 != -4096 && v50 != -8192 )
      sub_BD60C0(v49);
    return (_QWORD *)sub_C7D6A0(v48, v47, 8);
  }
  else
  {
    v38 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v53[0] = 2;
    v55 = 0;
    for ( m = &result[11 * v38]; m != result; result += 11 )
    {
      if ( result )
      {
        v40 = v53[0];
        result[2] = 0;
        result[3] = -4096;
        *result = &unk_49DDAE8;
        result[1] = v40 & 6;
        result[4] = v55;
      }
    }
  }
  return result;
}

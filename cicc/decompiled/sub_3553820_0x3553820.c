// Function: sub_3553820
// Address: 0x3553820
//
__int64 __fastcall sub_3553820(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 result; // rax
  __int64 v4; // rcx
  __int64 v5; // r13
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 *v9; // r13
  __int64 v10; // r9
  __int64 v11; // rdi
  int v12; // r11d
  _QWORD *v13; // r10
  unsigned int v14; // ecx
  _QWORD *v15; // rdx
  __int64 v16; // rax
  unsigned int v17; // esi
  __int64 v18; // r12
  int v19; // eax
  int v20; // ecx
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rsi
  int v24; // edx
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rcx
  unsigned int v28; // eax
  __int64 v29; // rbx
  char v30; // al
  __int64 v31; // rax
  unsigned __int64 v32; // rdi
  int v33; // eax
  int v34; // eax
  int v35; // eax
  int v36; // eax
  int v37; // eax
  __int64 v38; // rsi
  unsigned int v39; // r15d
  _QWORD *v40; // rdi
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // rbx
  unsigned __int64 v44; // rdi
  int v45; // r11d
  unsigned int v46; // r11d
  __int64 v47; // [rsp+8h] [rbp-58h]
  __int64 v48; // [rsp+8h] [rbp-58h]
  __int64 v49; // [rsp+8h] [rbp-58h]
  __int64 v50; // [rsp+10h] [rbp-50h]
  __int64 v51; // [rsp+20h] [rbp-40h]

  v2 = *(_QWORD *)a2;
  result = 11LL * *(unsigned int *)(a2 + 8);
  v4 = *(_QWORD *)a2 + 88LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v4 )
  {
    do
    {
      v50 = v2;
      v2 += 88;
      if ( v2 == v4 )
        return result;
      v5 = v2;
      v51 = v2 - 88;
      do
      {
        while ( 1 )
        {
          v6 = *(__int64 **)(v5 + 32);
          result = *(unsigned int *)(*v6 + 200);
          if ( *(_DWORD *)(**(_QWORD **)(v2 - 56) + 200LL) == (_DWORD)result )
            break;
          v5 += 88;
          if ( v5 == v4 )
            goto LABEL_49;
        }
        if ( *(_DWORD *)(v5 + 52) - *(_DWORD *)(v2 - 36) > 0 )
        {
          *(_DWORD *)(v2 - 36) = *(_DWORD *)(v5 + 52);
          v6 = *(__int64 **)(v5 + 32);
        }
        v7 = *(unsigned int *)(v5 + 40);
        if ( &v6[v7] != v6 )
        {
          v8 = v5;
          v9 = &v6[v7];
          while ( 1 )
          {
            v17 = *(_DWORD *)(v2 - 64);
            v18 = *v6;
            if ( !v17 )
              break;
            v10 = v17 - 1;
            v11 = *(_QWORD *)(v2 - 80);
            v12 = 1;
            v13 = 0;
            v14 = v10 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v15 = (_QWORD *)(v11 + 8LL * v14);
            v16 = *v15;
            if ( v18 == *v15 )
            {
LABEL_11:
              if ( v9 == ++v6 )
                goto LABEL_21;
            }
            else
            {
              while ( v16 != -4096 )
              {
                if ( v13 || v16 != -8192 )
                  v15 = v13;
                v14 = v10 & (v12 + v14);
                v16 = *(_QWORD *)(v11 + 8LL * v14);
                if ( v18 == v16 )
                  goto LABEL_11;
                ++v12;
                v13 = v15;
                v15 = (_QWORD *)(v11 + 8LL * v14);
              }
              v35 = *(_DWORD *)(v2 - 72);
              if ( !v13 )
                v13 = v15;
              ++*(_QWORD *)(v2 - 88);
              v24 = v35 + 1;
              if ( 4 * (v35 + 1) < 3 * v17 )
              {
                if ( v17 - *(_DWORD *)(v2 - 68) - v24 <= v17 >> 3 )
                {
                  v48 = v8;
                  sub_3553650(v51, v17);
                  v36 = *(_DWORD *)(v2 - 64);
                  if ( !v36 )
                  {
LABEL_67:
                    ++*(_DWORD *)(v50 + 16);
                    BUG();
                  }
                  v37 = v36 - 1;
                  v38 = *(_QWORD *)(v2 - 80);
                  v10 = 1;
                  v39 = v37 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
                  v8 = v48;
                  v13 = (_QWORD *)(v38 + 8LL * v39);
                  v24 = *(_DWORD *)(v2 - 72) + 1;
                  v40 = 0;
                  v41 = *v13;
                  if ( v18 != *v13 )
                  {
                    while ( v41 != -4096 )
                    {
                      if ( !v40 && v41 == -8192 )
                        v40 = v13;
                      v46 = v10 + 1;
                      v10 = v37 & (v39 + (unsigned int)v10);
                      v13 = (_QWORD *)(v38 + 8LL * (unsigned int)v10);
                      v39 = v10;
                      v41 = *v13;
                      if ( v18 == *v13 )
                        goto LABEL_16;
                      v10 = v46;
                    }
                    if ( v40 )
                      v13 = v40;
                  }
                }
                goto LABEL_16;
              }
LABEL_14:
              v47 = v8;
              sub_3553650(v51, 2 * v17);
              v19 = *(_DWORD *)(v2 - 64);
              if ( !v19 )
                goto LABEL_67;
              v20 = v19 - 1;
              v21 = *(_QWORD *)(v2 - 80);
              v8 = v47;
              LODWORD(v22) = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
              v13 = (_QWORD *)(v21 + 8LL * (unsigned int)v22);
              v23 = *v13;
              v24 = *(_DWORD *)(v2 - 72) + 1;
              if ( v18 != *v13 )
              {
                v45 = 1;
                v10 = 0;
                while ( v23 != -4096 )
                {
                  if ( !v10 && v23 == -8192 )
                    v10 = (__int64)v13;
                  v22 = v20 & (unsigned int)(v22 + v45);
                  v13 = (_QWORD *)(v21 + 8 * v22);
                  v23 = *v13;
                  if ( v18 == *v13 )
                    goto LABEL_16;
                  ++v45;
                }
                if ( v10 )
                  v13 = (_QWORD *)v10;
              }
LABEL_16:
              *(_DWORD *)(v2 - 72) = v24;
              if ( *v13 != -4096 )
                --*(_DWORD *)(v2 - 68);
              *v13 = v18;
              v25 = *(unsigned int *)(v2 - 48);
              if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(v2 - 44) )
              {
                v49 = v8;
                sub_C8D5F0(v2 - 56, (const void *)(v2 - 40), v25 + 1, 8u, v8, v10);
                v25 = *(unsigned int *)(v2 - 48);
                v8 = v49;
              }
              ++v6;
              *(_QWORD *)(*(_QWORD *)(v2 - 56) + 8 * v25) = v18;
              ++*(_DWORD *)(v2 - 48);
              if ( v9 == v6 )
              {
LABEL_21:
                v5 = v8;
                goto LABEL_22;
              }
            }
          }
          ++*(_QWORD *)(v2 - 88);
          goto LABEL_14;
        }
LABEL_22:
        v26 = v5 + 120;
        v27 = *(_QWORD *)a2;
        v28 = *(_DWORD *)(a2 + 8);
        v29 = 0x2E8BA2E8BA2E8BA3LL * ((*(_QWORD *)a2 + 88LL * v28 - (v5 + 88)) >> 3);
        if ( *(_QWORD *)a2 + 88LL * v28 - (v5 + 88) > 0 )
        {
          do
          {
            sub_C7D6A0(*(_QWORD *)(v26 - 112), 8LL * *(unsigned int *)(v26 - 96), 8);
            v31 = *(_QWORD *)(v26 - 24);
            ++*(_QWORD *)(v26 - 120);
            ++*(_QWORD *)(v26 - 32);
            *(_QWORD *)(v26 - 112) = v31;
            LODWORD(v31) = *(_DWORD *)(v26 - 16);
            *(_QWORD *)(v26 - 24) = 0;
            *(_DWORD *)(v26 - 104) = v31;
            LODWORD(v31) = *(_DWORD *)(v26 - 12);
            *(_DWORD *)(v26 - 16) = 0;
            *(_DWORD *)(v26 - 100) = v31;
            LODWORD(v31) = *(_DWORD *)(v26 - 8);
            *(_DWORD *)(v26 - 12) = 0;
            *(_DWORD *)(v26 - 96) = v31;
            LODWORD(v31) = *(_DWORD *)(v26 + 8);
            *(_DWORD *)(v26 - 8) = 0;
            if ( (_DWORD)v31 )
            {
              v32 = *(_QWORD *)(v26 - 88);
              if ( v32 != v26 - 72 )
                _libc_free(v32);
              *(_QWORD *)(v26 - 88) = *(_QWORD *)v26;
              v33 = *(_DWORD *)(v26 + 8);
              *(_DWORD *)(v26 + 8) = 0;
              *(_DWORD *)(v26 - 80) = v33;
              v34 = *(_DWORD *)(v26 + 12);
              *(_DWORD *)(v26 + 12) = 0;
              *(_DWORD *)(v26 - 76) = v34;
              *(_QWORD *)v26 = v26 + 16;
            }
            else
            {
              *(_DWORD *)(v26 - 80) = 0;
            }
            v30 = *(_BYTE *)(v26 + 16);
            v26 += 88;
            *(_BYTE *)(v26 - 160) = v30;
            *(_DWORD *)(v26 - 156) = *(_DWORD *)(v26 - 68);
            *(_DWORD *)(v26 - 152) = *(_DWORD *)(v26 - 64);
            *(_DWORD *)(v26 - 148) = *(_DWORD *)(v26 - 60);
            *(_DWORD *)(v26 - 144) = *(_DWORD *)(v26 - 56);
            *(_QWORD *)(v26 - 136) = *(_QWORD *)(v26 - 48);
            *(_DWORD *)(v26 - 128) = *(_DWORD *)(v26 - 40);
            --v29;
          }
          while ( v29 );
          v28 = *(_DWORD *)(a2 + 8);
          v27 = *(_QWORD *)a2;
        }
        v42 = v28 - 1;
        *(_DWORD *)(a2 + 8) = v42;
        v43 = v27 + 88 * v42;
        v44 = *(_QWORD *)(v43 + 32);
        if ( v44 != v43 + 48 )
          _libc_free(v44);
        sub_C7D6A0(*(_QWORD *)(v43 + 8), 8LL * *(unsigned int *)(v43 + 24), 8);
        result = *(_QWORD *)a2;
        v4 = *(_QWORD *)a2 + 88LL * *(unsigned int *)(a2 + 8);
      }
      while ( v5 != v4 );
LABEL_49:
      ;
    }
    while ( v2 != v4 );
  }
  return result;
}

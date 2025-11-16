// Function: sub_2E45F70
// Address: 0x2e45f70
//
void __fastcall sub_2E45F70(__int64 a1, int a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v10; // ecx
  int *v11; // r13
  int v12; // edi
  unsigned int *v13; // r9
  unsigned int *i; // r14
  __int64 v15; // rsi
  __int64 v16; // rcx
  unsigned int v17; // r14d
  __int64 v18; // r8
  int v19; // esi
  __int16 *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r9
  unsigned int v23; // ecx
  int *v24; // rdx
  int v25; // r11d
  int v26; // edx
  int v27; // r15d
  __int16 *v28; // r12
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned int v31; // esi
  int *v32; // rcx
  int v33; // edi
  int v34; // edx
  unsigned __int64 v35; // rdi
  int v36; // edx
  unsigned int v37; // r8d
  int *v38; // rdx
  int v39; // r10d
  int v40; // edx
  __int16 *v41; // rdi
  int v42; // ecx
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rsi
  int v46; // edx
  int v47; // r15d
  char *v48; // rsi
  __int64 v49; // r8
  char *v50; // rdx
  int v51; // eax
  char *v52; // rdi
  int v53; // r8d
  unsigned __int64 v54; // rdi
  int v55; // ecx
  int v56; // r9d
  int v57; // r9d
  int v58; // [rsp+4h] [rbp-5Ch]
  int *v60; // [rsp+8h] [rbp-58h]
  int *v61; // [rsp+8h] [rbp-58h]
  int *v62; // [rsp+8h] [rbp-58h]
  _QWORD v63[10]; // [rsp+10h] [rbp-50h] BYREF

  v6 = *(unsigned int *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)v6 )
  {
    v10 = (v6 - 1) & (37 * a2);
    v11 = (int *)(v7 + ((unsigned __int64)v10 << 7));
    v12 = *v11;
    if ( *v11 == a2 )
    {
LABEL_3:
      if ( v11 != (int *)(v7 + (v6 << 7)) )
      {
        v13 = (unsigned int *)*((_QWORD *)v11 + 11);
        for ( i = &v13[v11[24]]; i != v13; ++v13 )
        {
          v43 = *(_QWORD *)(a3 + 8);
          v42 = *(_DWORD *)(v43 + 24LL * *v13 + 16) & 0xFFF;
          v41 = (__int16 *)(*(_QWORD *)(a3 + 56) + 2LL * (*(_DWORD *)(v43 + 24LL * *v13 + 16) >> 12));
          do
          {
            if ( !v41 )
              break;
            v44 = *(unsigned int *)(a1 + 24);
            v45 = *(_QWORD *)(a1 + 8);
            if ( (_DWORD)v44 )
            {
              v37 = (v44 - 1) & (37 * v42);
              v38 = (int *)(v45 + ((unsigned __int64)v37 << 7));
              v39 = *v38;
              if ( *v38 == v42 )
              {
LABEL_28:
                if ( v38 != (int *)(v45 + (v44 << 7)) )
                  *((_BYTE *)v38 + 120) = 0;
              }
              else
              {
                v36 = 1;
                while ( v39 != -1 )
                {
                  v37 = (v44 - 1) & (v36 + v37);
                  v58 = v36 + 1;
                  v38 = (int *)(v45 + ((unsigned __int64)v37 << 7));
                  v39 = *v38;
                  if ( v42 == *v38 )
                    goto LABEL_28;
                  v36 = v58;
                }
              }
            }
            v40 = *v41++;
            v42 += v40;
          }
          while ( (_WORD)v40 );
        }
        v15 = *((_QWORD *)v11 + 1);
        if ( v15 )
        {
          sub_2E44C10((__int64)v63, v15, a4, a5);
          v16 = *(_QWORD *)(a3 + 8);
          v17 = *(_DWORD *)(v63[0] + 8LL);
          v18 = *(unsigned int *)(v63[1] + 8LL);
          v19 = *(_DWORD *)(v16 + 24LL * v17 + 16) & 0xFFF;
          v20 = (__int16 *)(*(_QWORD *)(a3 + 56) + 2LL * (*(_DWORD *)(v16 + 24LL * v17 + 16) >> 12));
          do
          {
            if ( !v20 )
              break;
            v21 = *(unsigned int *)(a1 + 24);
            v22 = *(_QWORD *)(a1 + 8);
            if ( (_DWORD)v21 )
            {
              v23 = (v21 - 1) & (37 * v19);
              v24 = (int *)(v22 + ((unsigned __int64)v23 << 7));
              v25 = *v24;
              if ( *v24 == v19 )
              {
LABEL_10:
                if ( v24 != (int *)(v22 + (v21 << 7)) )
                  *((_BYTE *)v24 + 120) = 0;
              }
              else
              {
                v46 = 1;
                while ( v25 != -1 )
                {
                  v47 = v46 + 1;
                  v23 = (v21 - 1) & (v46 + v23);
                  v24 = (int *)(v22 + ((unsigned __int64)v23 << 7));
                  v25 = *v24;
                  if ( v19 == *v24 )
                    goto LABEL_10;
                  v46 = v47;
                }
              }
            }
            v26 = *v20++;
            v19 += v26;
          }
          while ( (_WORD)v26 );
          v27 = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 24 * v18 + 16) & 0xFFF;
          v28 = (__int16 *)(*(_QWORD *)(a3 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(a3 + 8) + 24 * v18 + 16) >> 12));
          do
          {
            if ( !v28 )
              break;
            v29 = *(unsigned int *)(a1 + 24);
            v30 = *(_QWORD *)(a1 + 8);
            if ( (_DWORD)v29 )
            {
              v31 = (v29 - 1) & (37 * v27);
              v32 = (int *)(v30 + ((unsigned __int64)v31 << 7));
              v33 = *v32;
              if ( v27 == *v32 )
              {
LABEL_17:
                if ( v32 != (int *)(v30 + (v29 << 7)) && *((_QWORD *)v32 + 2) )
                {
                  v48 = (char *)*((_QWORD *)v32 + 11);
                  v49 = (unsigned int)v32[24];
                  v50 = &v48[4 * v49];
                  while ( v48 != v50 )
                  {
                    v51 = *(_DWORD *)v48;
                    v52 = v48;
                    v48 += 4;
                    if ( v17 == v51 )
                    {
                      if ( v48 != v50 )
                      {
                        v60 = v32;
                        memmove(v52, v48, v50 - v48);
                        v32 = v60;
                        LODWORD(v49) = v60[24];
                      }
                      v53 = v49 - 1;
                      v32[24] = v53;
                      if ( !v53 && !*((_QWORD *)v32 + 1) )
                      {
                        v54 = *((_QWORD *)v32 + 11);
                        if ( (int *)v54 != v32 + 26 )
                        {
                          v61 = v32;
                          _libc_free(v54);
                          v32 = v61;
                        }
                        if ( !*((_BYTE *)v32 + 52) )
                        {
                          v62 = v32;
                          _libc_free(*((_QWORD *)v32 + 4));
                          v32 = v62;
                        }
                        *v32 = -2;
                        --*(_DWORD *)(a1 + 16);
                        ++*(_DWORD *)(a1 + 20);
                      }
                      break;
                    }
                  }
                }
              }
              else
              {
                v55 = 1;
                while ( v33 != -1 )
                {
                  v56 = v55 + 1;
                  v31 = (v29 - 1) & (v55 + v31);
                  v32 = (int *)(v30 + ((unsigned __int64)v31 << 7));
                  v33 = *v32;
                  if ( v27 == *v32 )
                    goto LABEL_17;
                  v55 = v56;
                }
              }
            }
            v34 = *v28++;
            v27 += v34;
          }
          while ( (_WORD)v34 );
        }
        v35 = *((_QWORD *)v11 + 11);
        if ( (int *)v35 != v11 + 26 )
          _libc_free(v35);
        if ( !*((_BYTE *)v11 + 52) )
          _libc_free(*((_QWORD *)v11 + 4));
        *v11 = -2;
        --*(_DWORD *)(a1 + 16);
        ++*(_DWORD *)(a1 + 20);
      }
    }
    else
    {
      v57 = 1;
      while ( v12 != -1 )
      {
        v10 = (v6 - 1) & (v57 + v10);
        v11 = (int *)(v7 + ((unsigned __int64)v10 << 7));
        v12 = *v11;
        if ( *v11 == a2 )
          goto LABEL_3;
        ++v57;
      }
    }
  }
}

// Function: sub_B2F150
// Address: 0xb2f150
//
__int64 __fastcall sub_B2F150(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  unsigned __int8 v5; // al
  _QWORD *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int8 v10; // r11
  unsigned int v11; // r14d
  bool v12; // al
  __int64 v13; // rcx
  __int64 v14; // rdx
  _QWORD *v15; // rbx
  unsigned int v16; // esi
  __int64 v17; // r8
  unsigned __int64 v18; // r15
  unsigned int v19; // edi
  _QWORD *v20; // rcx
  _QWORD *v21; // rdx
  _QWORD *v22; // r10
  int v23; // eax
  int v24; // edx
  int v25; // edx
  int v26; // ecx
  __int64 v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // rsi
  int v30; // r9d
  _QWORD *v31; // r8
  int v32; // eax
  int v33; // eax
  __int64 v34; // rsi
  int v35; // r8d
  unsigned int v36; // r15d
  _QWORD *v37; // rdi
  __int64 v38; // rcx
  int v39; // [rsp+14h] [rbp-3Ch]
  __int64 v40; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    v3 = sub_B91C10(a2, 2);
    v4 = v3;
    if ( v3 )
    {
      v40 = v3 - 16;
      v5 = *(_BYTE *)(v3 - 16);
      v6 = (v5 & 2) != 0 ? *(_QWORD **)(v4 - 32) : (_QWORD *)(v40 - 8LL * ((v5 >> 2) & 0xF));
      if ( !*(_BYTE *)*v6 )
      {
        v7 = sub_B91420(*v6, 2);
        if ( v8 == 20
          && !(*(_QWORD *)v7 ^ 0x6E6F6974636E7566LL | *(_QWORD *)(v7 + 8) ^ 0x635F7972746E655FLL)
          && *(_DWORD *)(v7 + 16) == 1953396079 )
        {
          v10 = *(_BYTE *)(v4 - 16);
          v11 = 2;
          v12 = (v10 & 2) != 0;
          while ( 1 )
          {
            if ( v12 )
            {
              if ( v11 >= *(_DWORD *)(v4 - 24) )
                return a1;
              v13 = *(_QWORD *)(v4 - 32);
            }
            else
            {
              if ( v11 >= ((*(_WORD *)(v4 - 16) >> 6) & 0xFu) )
                return a1;
              v13 = v40 - 8LL * ((v10 >> 2) & 0xF);
            }
            v14 = *(_QWORD *)(*(_QWORD *)(v13 + 8LL * v11) + 136LL);
            v15 = *(_QWORD **)(v14 + 24);
            if ( *(_DWORD *)(v14 + 32) > 0x40u )
              v15 = (_QWORD *)*v15;
            v16 = *(_DWORD *)(a1 + 24);
            if ( v16 )
            {
              v17 = *(_QWORD *)(a1 + 8);
              v18 = ((0xBF58476D1CE4E5B9LL * (unsigned __int64)v15) >> 31) ^ (0xBF58476D1CE4E5B9LL * (_QWORD)v15);
              v19 = v18 & (v16 - 1);
              v20 = (_QWORD *)(v17 + 8LL * v19);
              v21 = (_QWORD *)*v20;
              if ( v15 == (_QWORD *)*v20 )
                goto LABEL_18;
              v39 = 1;
              v22 = 0;
              while ( v21 != (_QWORD *)-1LL )
              {
                if ( v22 || v21 != (_QWORD *)-2LL )
                  v20 = v22;
                v19 = (v16 - 1) & (v39 + v19);
                v21 = *(_QWORD **)(v17 + 8LL * v19);
                if ( v15 == v21 )
                  goto LABEL_18;
                ++v39;
                v22 = v20;
                v20 = (_QWORD *)(v17 + 8LL * v19);
              }
              v23 = *(_DWORD *)(a1 + 16);
              if ( !v22 )
                v22 = v20;
              ++*(_QWORD *)a1;
              v24 = v23 + 1;
              if ( 4 * (v23 + 1) < 3 * v16 )
              {
                if ( v16 - *(_DWORD *)(a1 + 20) - v24 <= v16 >> 3 )
                {
                  sub_A32210(a1, v16);
                  v32 = *(_DWORD *)(a1 + 24);
                  if ( !v32 )
                  {
LABEL_60:
                    ++*(_DWORD *)(a1 + 16);
                    BUG();
                  }
                  v33 = v32 - 1;
                  v34 = *(_QWORD *)(a1 + 8);
                  v35 = 1;
                  v36 = v33 & v18;
                  v22 = (_QWORD *)(v34 + 8LL * v36);
                  v24 = *(_DWORD *)(a1 + 16) + 1;
                  v37 = 0;
                  v38 = *v22;
                  if ( v15 != (_QWORD *)*v22 )
                  {
                    while ( v38 != -1 )
                    {
                      if ( !v37 && v38 == -2 )
                        v37 = v22;
                      v36 = v33 & (v35 + v36);
                      v22 = (_QWORD *)(v34 + 8LL * v36);
                      v38 = *v22;
                      if ( v15 == (_QWORD *)*v22 )
                        goto LABEL_28;
                      ++v35;
                    }
                    if ( v37 )
                      v22 = v37;
                  }
                }
                goto LABEL_28;
              }
            }
            else
            {
              ++*(_QWORD *)a1;
            }
            sub_A32210(a1, 2 * v16);
            v25 = *(_DWORD *)(a1 + 24);
            if ( !v25 )
              goto LABEL_60;
            v26 = v25 - 1;
            v27 = *(_QWORD *)(a1 + 8);
            v28 = (v25 - 1) & (((0xBF58476D1CE4E5B9LL * (unsigned __int64)v15) >> 31) ^ (484763065 * (_DWORD)v15));
            v22 = (_QWORD *)(v27 + 8LL * v28);
            v29 = *v22;
            v24 = *(_DWORD *)(a1 + 16) + 1;
            if ( v15 != (_QWORD *)*v22 )
            {
              v30 = 1;
              v31 = 0;
              while ( v29 != -1 )
              {
                if ( v29 == -2 && !v31 )
                  v31 = v22;
                v28 = v26 & (v30 + v28);
                v22 = (_QWORD *)(v27 + 8LL * v28);
                v29 = *v22;
                if ( v15 == (_QWORD *)*v22 )
                  goto LABEL_28;
                ++v30;
              }
              if ( v31 )
                v22 = v31;
            }
LABEL_28:
            *(_DWORD *)(a1 + 16) = v24;
            if ( *v22 != -1 )
              --*(_DWORD *)(a1 + 20);
            *v22 = v15;
            v10 = *(_BYTE *)(v4 - 16);
            v12 = (v10 & 2) != 0;
LABEL_18:
            ++v11;
          }
        }
      }
    }
  }
  return a1;
}

// Function: sub_322B4B0
// Address: 0x322b4b0
//
__int64 __fastcall sub_322B4B0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rax
  unsigned __int8 v4; // dl
  __int64 v5; // rax
  unsigned __int8 *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r12
  __int64 v10; // r8
  unsigned int v11; // edi
  __int64 *v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rbx
  unsigned int v16; // esi
  int v17; // eax
  int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // edx
  int v21; // ecx
  __int64 *v22; // rax
  __int64 v23; // rdi
  unsigned __int8 v24; // al
  __int64 v25; // rbx
  int v26; // r10d
  int v27; // ecx
  int v28; // eax
  int v29; // edx
  __int64 v30; // rdi
  __int64 *v31; // r8
  unsigned int v32; // r15d
  int v33; // r9d
  __int64 v34; // rsi
  int v35; // r10d
  __int64 *v36; // r9
  __int64 v37; // [rsp-68h] [rbp-68h]
  __int64 v38; // [rsp-60h] [rbp-60h]
  __int64 v39; // [rsp-58h] [rbp-58h]
  __int64 v40; // [rsp-50h] [rbp-50h]
  __int64 v41[8]; // [rsp-40h] [rbp-40h] BYREF

  result = (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL) + 544LL) - 42);
  if ( (unsigned int)result <= 1 && *(_DWORD *)(a1 + 6224) == 1 )
  {
    v3 = sub_B92180(**(_QWORD **)(a1 + 3048));
    v4 = *(_BYTE *)(v3 - 16);
    v5 = (v4 & 2) != 0 ? *(_QWORD *)(v3 - 32) : v3 - 16 - 8LL * ((v4 >> 2) & 0xF);
    result = *(_QWORD *)(v5 + 40);
    if ( *(_DWORD *)(result + 32) == 3 )
    {
      v6 = (unsigned __int8 *)sub_C94E20((__int64)qword_4F863B0);
      result = v6 ? *v6 : LOBYTE(qword_4F863B0[2]);
      if ( (_BYTE)result )
      {
        v7 = *(_QWORD *)(a1 + 3048);
        v8 = *(_QWORD *)(v7 + 328);
        result = v7 + 320;
        v37 = result;
        v39 = v8;
        if ( v8 != result )
        {
          v38 = a1 + 6232;
          while ( 1 )
          {
            v9 = *(_QWORD *)(v39 + 56);
            v40 = v39 + 48;
            if ( v9 != v39 + 48 )
              break;
LABEL_38:
            result = *(_QWORD *)(v39 + 8);
            v39 = result;
            if ( v37 == result )
              return result;
          }
          while ( 1 )
          {
            v14 = *(_QWORD *)(v9 + 56);
            v41[0] = v14;
            if ( v14 )
            {
              sub_B96E90((__int64)v41, v14, 1);
              if ( v41[0] )
              {
                v15 = sub_B10D40((__int64)v41);
                if ( v15 )
                {
                  while ( 1 )
                  {
                    v16 = *(_DWORD *)(a1 + 6256);
                    if ( !v16 )
                      break;
                    v10 = *(_QWORD *)(a1 + 6240);
                    v11 = (v16 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
                    v12 = (__int64 *)(v10 + 8LL * v11);
                    v13 = *v12;
                    if ( *v12 == v15 )
                      goto LABEL_16;
                    v26 = 1;
                    v22 = 0;
                    while ( v13 != -4096 )
                    {
                      if ( v22 || v13 != -8192 )
                        v12 = v22;
                      v11 = (v16 - 1) & (v26 + v11);
                      v13 = *(_QWORD *)(v10 + 8LL * v11);
                      if ( v13 == v15 )
                        goto LABEL_16;
                      v22 = v12;
                      ++v26;
                      v12 = (__int64 *)(v10 + 8LL * v11);
                    }
                    if ( !v22 )
                      v22 = v12;
                    v27 = *(_DWORD *)(a1 + 6248);
                    ++*(_QWORD *)(a1 + 6232);
                    v21 = v27 + 1;
                    if ( 4 * v21 >= 3 * v16 )
                      goto LABEL_25;
                    if ( v16 - *(_DWORD *)(a1 + 6252) - v21 <= v16 >> 3 )
                    {
                      sub_322A7C0(v38, v16);
                      v28 = *(_DWORD *)(a1 + 6256);
                      if ( !v28 )
                      {
LABEL_75:
                        ++*(_DWORD *)(a1 + 6248);
                        BUG();
                      }
                      v29 = v28 - 1;
                      v30 = *(_QWORD *)(a1 + 6240);
                      v31 = 0;
                      v32 = (v28 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
                      v33 = 1;
                      v21 = *(_DWORD *)(a1 + 6248) + 1;
                      v22 = (__int64 *)(v30 + 8LL * v32);
                      v34 = *v22;
                      if ( *v22 != v15 )
                      {
                        while ( v34 != -4096 )
                        {
                          if ( v34 == -8192 && !v31 )
                            v31 = v22;
                          v32 = v29 & (v33 + v32);
                          v22 = (__int64 *)(v30 + 8LL * v32);
                          v34 = *v22;
                          if ( *v22 == v15 )
                            goto LABEL_27;
                          ++v33;
                        }
                        if ( v31 )
                          v22 = v31;
                      }
                    }
LABEL_27:
                    *(_DWORD *)(a1 + 6248) = v21;
                    if ( *v22 != -4096 )
                      --*(_DWORD *)(a1 + 6252);
                    *v22 = v15;
                    v24 = *(_BYTE *)(v15 - 16);
                    if ( (v24 & 2) != 0 )
                    {
                      if ( *(_DWORD *)(v15 - 24) != 2 )
                        goto LABEL_16;
                      v25 = *(_QWORD *)(v15 - 32);
                    }
                    else
                    {
                      if ( ((*(_WORD *)(v15 - 16) >> 6) & 0xF) != 2 )
                        goto LABEL_16;
                      v25 = v15 - 16 - 8LL * ((v24 >> 2) & 0xF);
                    }
                    v15 = *(_QWORD *)(v25 + 8);
                    if ( !v15 )
                      goto LABEL_16;
                  }
                  ++*(_QWORD *)(a1 + 6232);
LABEL_25:
                  sub_322A7C0(v38, 2 * v16);
                  v17 = *(_DWORD *)(a1 + 6256);
                  if ( !v17 )
                    goto LABEL_75;
                  v18 = v17 - 1;
                  v19 = *(_QWORD *)(a1 + 6240);
                  v20 = (v17 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
                  v21 = *(_DWORD *)(a1 + 6248) + 1;
                  v22 = (__int64 *)(v19 + 8LL * v20);
                  v23 = *v22;
                  if ( *v22 != v15 )
                  {
                    v35 = 1;
                    v36 = 0;
                    while ( v23 != -4096 )
                    {
                      if ( v23 == -8192 && !v36 )
                        v36 = v22;
                      v20 = v18 & (v35 + v20);
                      v22 = (__int64 *)(v19 + 8LL * v20);
                      v23 = *v22;
                      if ( *v22 == v15 )
                        goto LABEL_27;
                      ++v35;
                    }
                    if ( v36 )
                      v22 = v36;
                  }
                  goto LABEL_27;
                }
LABEL_16:
                if ( v41[0] )
                  sub_B91220((__int64)v41, v41[0]);
              }
            }
            if ( (*(_BYTE *)v9 & 4) != 0 )
            {
              v9 = *(_QWORD *)(v9 + 8);
              if ( v40 == v9 )
                goto LABEL_38;
            }
            else
            {
              while ( (*(_BYTE *)(v9 + 44) & 8) != 0 )
                v9 = *(_QWORD *)(v9 + 8);
              v9 = *(_QWORD *)(v9 + 8);
              if ( v40 == v9 )
                goto LABEL_38;
            }
          }
        }
      }
    }
  }
  return result;
}

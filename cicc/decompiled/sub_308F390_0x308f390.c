// Function: sub_308F390
// Address: 0x308f390
//
__int64 __fastcall sub_308F390(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rdx
  int v10; // r14d
  __int64 v11; // rdx
  __int64 v12; // rsi
  int v13; // r15d
  unsigned int v14; // r8d
  _DWORD *v15; // rcx
  int v16; // edi
  unsigned int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // edi
  _DWORD *v20; // rcx
  int v21; // edx
  _DWORD *v22; // r10
  int v23; // edi
  int v24; // edx
  int v25; // r10d
  int v26; // r9d
  __int64 v27; // r8
  _DWORD *v28; // rcx
  unsigned int v29; // r15d
  int v30; // esi
  int v31; // edi
  int v32; // ecx
  int v33; // r10d
  int v34; // r10d
  int v35; // r9d
  __int64 v36; // r8
  unsigned int v37; // r15d
  int v38; // edi
  int v39; // esi
  __int64 v40; // [rsp+0h] [rbp-50h]
  __int64 v41; // [rsp+8h] [rbp-48h]
  _DWORD *v42; // [rsp+10h] [rbp-40h]
  int v43; // [rsp+10h] [rbp-40h]

  result = *(unsigned __int16 *)(*(_QWORD *)(a2 + 16) + 2LL);
  if ( (_WORD)result )
  {
    v7 = 0;
    v8 = 40 * result;
    result = a3;
    do
    {
      v9 = v7 + *(_QWORD *)(a2 + 32);
      if ( !*(_BYTE *)v9 && (*(_BYTE *)(v9 + 3) & 0x10) != 0 )
      {
        v10 = *(_DWORD *)(v9 + 8);
        v11 = *(unsigned int *)(result + 24);
        v12 = *(_QWORD *)(result + 8);
        if ( (_DWORD)v11 )
        {
          v13 = 37 * v10;
          v14 = (v11 - 1) & (37 * v10);
          v15 = (_DWORD *)(v12 + 8LL * v14);
          v16 = *v15;
          if ( v10 != *v15 )
          {
            v32 = 1;
            while ( v16 != -1 )
            {
              v33 = v32 + 1;
              v14 = (v11 - 1) & (v32 + v14);
              v15 = (_DWORD *)(v12 + 8LL * v14);
              v16 = *v15;
              if ( v10 == *v15 )
                goto LABEL_8;
              v32 = v33;
            }
            goto LABEL_3;
          }
LABEL_8:
          if ( v15 != (_DWORD *)(v12 + 8 * v11) )
          {
            v40 = a4;
            v41 = result;
            v42 = v15;
            sub_308E6E0(a1, v15[1]);
            a4 = v40;
            result = v41;
            v42[1] = -1;
            v17 = *(_DWORD *)(v40 + 24);
            if ( !v17 )
            {
              ++*(_QWORD *)v40;
              goto LABEL_29;
            }
            v18 = *(_QWORD *)(v40 + 8);
            v19 = (v17 - 1) & v13;
            v20 = (_DWORD *)(v18 + 4LL * v19);
            v21 = *v20;
            if ( v10 != *v20 )
            {
              v43 = 1;
              v22 = 0;
              while ( v21 != -1 )
              {
                if ( v21 != -2 || v22 )
                  v20 = v22;
                v19 = (v17 - 1) & (v43 + v19);
                v21 = *(_DWORD *)(v18 + 4LL * v19);
                if ( v10 == v21 )
                  goto LABEL_3;
                ++v43;
                v22 = v20;
                v20 = (_DWORD *)(v18 + 4LL * v19);
              }
              v23 = *(_DWORD *)(v40 + 16);
              if ( !v22 )
                v22 = v20;
              ++*(_QWORD *)v40;
              v24 = v23 + 1;
              if ( 4 * (v23 + 1) < 3 * v17 )
              {
                if ( v17 - *(_DWORD *)(v40 + 20) - v24 <= v17 >> 3 )
                {
                  sub_A08C50(v40, v17);
                  a4 = v40;
                  v25 = *(_DWORD *)(v40 + 24);
                  if ( !v25 )
                    goto LABEL_51;
                  v26 = v25 - 1;
                  v27 = *(_QWORD *)(v40 + 8);
                  v28 = 0;
                  v29 = (v25 - 1) & v13;
                  v30 = 1;
                  v22 = (_DWORD *)(v27 + 4LL * v29);
                  v24 = *(_DWORD *)(v40 + 16) + 1;
                  result = v41;
                  v31 = *v22;
                  if ( v10 != *v22 )
                  {
                    while ( v31 != -1 )
                    {
                      if ( v31 == -2 && !v28 )
                        v28 = v22;
                      v29 = v26 & (v30 + v29);
                      v22 = (_DWORD *)(v27 + 4LL * v29);
                      v31 = *v22;
                      if ( v10 == *v22 )
                        goto LABEL_31;
                      ++v30;
                    }
                    goto LABEL_20;
                  }
                }
                goto LABEL_31;
              }
LABEL_29:
              sub_A08C50(v40, 2 * v17);
              a4 = v40;
              v34 = *(_DWORD *)(v40 + 24);
              if ( !v34 )
              {
LABEL_51:
                ++*(_DWORD *)(a4 + 16);
                BUG();
              }
              v35 = v34 - 1;
              v36 = *(_QWORD *)(v40 + 8);
              v37 = (v34 - 1) & v13;
              v22 = (_DWORD *)(v36 + 4LL * v37);
              v24 = *(_DWORD *)(v40 + 16) + 1;
              result = v41;
              v38 = *v22;
              if ( v10 != *v22 )
              {
                v39 = 1;
                v28 = 0;
                while ( v38 != -1 )
                {
                  if ( v38 == -2 && !v28 )
                    v28 = v22;
                  v37 = v35 & (v39 + v37);
                  v22 = (_DWORD *)(v36 + 4LL * v37);
                  v38 = *v22;
                  if ( v10 == *v22 )
                    goto LABEL_31;
                  ++v39;
                }
LABEL_20:
                if ( v28 )
                  v22 = v28;
              }
LABEL_31:
              *(_DWORD *)(a4 + 16) = v24;
              if ( *v22 != -1 )
                --*(_DWORD *)(a4 + 20);
              *v22 = v10;
            }
          }
        }
      }
LABEL_3:
      v7 += 40;
    }
    while ( v8 != v7 );
  }
  return result;
}

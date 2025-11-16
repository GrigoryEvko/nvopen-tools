// Function: sub_307C930
// Address: 0x307c930
//
__int64 __fastcall sub_307C930(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // esi
  __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r8
  __int64 result; // rax
  __int64 v12; // r12
  __int64 v13; // r11
  int v14; // r14d
  unsigned int v15; // ebx
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // r13d
  unsigned int v19; // esi
  __int64 v20; // rdx
  unsigned int v21; // r8d
  unsigned int v22; // ecx
  int v23; // edi
  unsigned int v24; // ecx
  __int64 v25; // rdx
  __int64 v26; // rax
  int v27; // r10d
  _DWORD *v28; // r10
  __int64 v29; // rdi
  int v30; // eax
  int v31; // eax
  int v32; // eax
  int v33; // eax
  __int64 v34; // r8
  unsigned int v35; // edi
  int v36; // esi
  int v37; // ecx
  _DWORD *v38; // rdx
  int v39; // eax
  __int64 v40; // r8
  _DWORD *v41; // rcx
  int v42; // esi
  unsigned int v43; // edx
  int v44; // edi
  int v45; // r9d
  int v46; // r9d
  __int64 v47; // rsi
  __int64 v48; // [rsp+8h] [rbp-58h]
  __int64 v49; // [rsp+8h] [rbp-58h]
  unsigned int v50; // [rsp+10h] [rbp-50h]
  __int64 v51; // [rsp+18h] [rbp-48h]
  __int64 v52; // [rsp+20h] [rbp-40h]
  __int64 v53; // [rsp+28h] [rbp-38h]
  __int64 v54; // [rsp+28h] [rbp-38h]
  int i; // [rsp+28h] [rbp-38h]
  int v56; // [rsp+28h] [rbp-38h]
  int v57; // [rsp+28h] [rbp-38h]
  int v58; // [rsp+28h] [rbp-38h]

  v6 = *(_DWORD *)(a1 + 136);
  v7 = *(_QWORD *)(a1 + 120);
  if ( v6 )
  {
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
      goto LABEL_3;
    v32 = 1;
    while ( v10 != -4096 )
    {
      v45 = v32 + 1;
      v8 = (v6 - 1) & (v32 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      v32 = v45;
    }
  }
  v9 = (__int64 *)(v7 + 16LL * v6);
LABEL_3:
  v53 = a2;
  v51 = v9[1];
  result = sub_2E311E0(a3);
  v12 = *(_QWORD *)(a3 + 56);
  v52 = result;
  if ( v12 != result )
  {
    v13 = v53;
    do
    {
      v54 = v13;
      result = sub_2E88FE0(v12);
      v13 = v54;
      v14 = *(_DWORD *)(v12 + 40) & 0xFFFFFF;
      v15 = result + *(unsigned __int8 *)(*(_QWORD *)(v12 + 16) + 9LL);
      while ( v14 != v15 )
      {
        while ( 1 )
        {
          v16 = v15;
          v17 = *(_QWORD *)(v12 + 32);
          ++v15;
          result = v17 + 40 * v16;
          if ( !*(_BYTE *)result )
          {
            v18 = *(_DWORD *)(result + 8);
            if ( v18 < 0 )
            {
              result = 5LL * v15;
              if ( v13 == *(_QWORD *)(v17 + 40LL * v15 + 24) )
              {
                v19 = *(_DWORD *)(a1 + 80);
                v20 = *(_QWORD *)(a1 + 64);
                if ( v19 )
                  break;
              }
            }
          }
LABEL_7:
          if ( v14 == v15 )
            goto LABEL_15;
        }
        v21 = v19 - 1;
        v22 = (v19 - 1) & (37 * v18);
        result = v20 + 8LL * v22;
        v23 = *(_DWORD *)result;
        if ( v18 == *(_DWORD *)result )
        {
LABEL_13:
          v24 = *(_DWORD *)(result + 4);
          v25 = 1LL << v24;
          v26 = 8LL * (v24 >> 6);
        }
        else
        {
          v50 = (v19 - 1) & (37 * v18);
          v27 = *(_DWORD *)result;
          for ( i = 1; ; ++i )
          {
            if ( v27 == -1 )
              goto LABEL_7;
            result = (unsigned int)(i + 1);
            v50 = v21 & (v50 + i);
            v27 = *(_DWORD *)(v20 + 8LL * v50);
            if ( v18 == v27 )
              break;
          }
          v56 = 1;
          result = v20 + 8LL * (v21 & (37 * v18));
          v28 = 0;
          while ( v23 != -1 )
          {
            if ( v23 == -2 && !v28 )
              v28 = (_DWORD *)result;
            v22 = v21 & (v56 + v22);
            result = v20 + 8LL * v22;
            v23 = *(_DWORD *)result;
            if ( v18 == *(_DWORD *)result )
              goto LABEL_13;
            ++v56;
          }
          v29 = a1 + 56;
          if ( !v28 )
            v28 = (_DWORD *)result;
          v30 = *(_DWORD *)(a1 + 72);
          ++*(_QWORD *)(a1 + 56);
          v31 = v30 + 1;
          if ( 4 * v31 >= 3 * v19 )
          {
            v48 = v13;
            sub_2E518D0(v29, 2 * v19);
            v33 = *(_DWORD *)(a1 + 80);
            if ( !v33 )
              goto LABEL_67;
            v34 = *(_QWORD *)(a1 + 64);
            v57 = v33 - 1;
            v13 = v48;
            v35 = (v33 - 1) & (37 * v18);
            v28 = (_DWORD *)(v34 + 8LL * v35);
            v36 = *v28;
            v31 = *(_DWORD *)(a1 + 72) + 1;
            if ( v18 != *v28 )
            {
              v37 = 1;
              v38 = 0;
              while ( v36 != -1 )
              {
                if ( v36 == -2 && !v38 )
                  v38 = v28;
                v35 = (v35 + v37) & v57;
                v28 = (_DWORD *)(v34 + 8LL * v35);
                v36 = *v28;
                if ( v18 == *v28 )
                  goto LABEL_30;
                ++v37;
              }
              if ( v38 )
                v28 = v38;
            }
          }
          else if ( v19 - *(_DWORD *)(a1 + 76) - v31 <= v19 >> 3 )
          {
            v49 = v13;
            sub_2E518D0(v29, v19);
            v39 = *(_DWORD *)(a1 + 80);
            if ( !v39 )
            {
LABEL_67:
              ++*(_DWORD *)(a1 + 72);
              BUG();
            }
            v40 = *(_QWORD *)(a1 + 64);
            v41 = 0;
            v58 = v39 - 1;
            v13 = v49;
            v42 = 1;
            v43 = (v39 - 1) & (37 * v18);
            v28 = (_DWORD *)(v40 + 8LL * v43);
            v44 = *v28;
            v31 = *(_DWORD *)(a1 + 72) + 1;
            if ( v18 != *v28 )
            {
              while ( v44 != -1 )
              {
                if ( v44 == -2 && !v41 )
                  v41 = v28;
                v46 = v42 + 1;
                v47 = v58 & (v43 + v42);
                v28 = (_DWORD *)(v40 + 8 * v47);
                v43 = v47;
                v44 = *v28;
                if ( v18 == *v28 )
                  goto LABEL_30;
                v42 = v46;
              }
              if ( v41 )
                v28 = v41;
            }
          }
LABEL_30:
          *(_DWORD *)(a1 + 72) = v31;
          if ( *v28 != -1 )
            --*(_DWORD *)(a1 + 76);
          *v28 = v18;
          v25 = 1;
          v26 = 0;
          v28[1] = 0;
        }
        result = *(_QWORD *)(v51 + 96) + v26;
        *(_QWORD *)result |= v25;
      }
LABEL_15:
      if ( (*(_BYTE *)v12 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v12 + 44) & 8) != 0 )
          v12 = *(_QWORD *)(v12 + 8);
      }
      v12 = *(_QWORD *)(v12 + 8);
    }
    while ( v52 != v12 );
  }
  return result;
}

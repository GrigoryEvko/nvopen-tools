// Function: sub_21EAF50
// Address: 0x21eaf50
//
__int64 __fastcall sub_21EAF50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // r9
  __int64 result; // rax
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // rbx
  int v15; // r13d
  unsigned int v16; // esi
  __int64 v17; // r9
  unsigned int v18; // r8d
  int *v19; // rax
  int v20; // edi
  unsigned int v21; // ecx
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // r11d
  int *v25; // rdx
  int v26; // eax
  int v27; // eax
  int v28; // ecx
  int v29; // ecx
  __int64 v30; // r8
  __int64 v31; // rsi
  int v32; // edi
  int v33; // r10d
  int *v34; // r9
  int v35; // esi
  int v36; // esi
  __int64 v37; // r8
  int v38; // r10d
  __int64 v39; // rcx
  int v40; // edi
  int v41; // eax
  int v42; // edx
  int *v43; // r11
  int v44; // r10d
  int v45; // [rsp+0h] [rbp-50h]
  __int64 v46; // [rsp+8h] [rbp-48h]
  __int64 v47; // [rsp+10h] [rbp-40h]
  __int64 v48; // [rsp+18h] [rbp-38h]

  v5 = *(unsigned int *)(a1 + 136);
  v6 = *(_QWORD *)(a1 + 120);
  if ( (_DWORD)v5 )
  {
    v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
      goto LABEL_3;
    v41 = 1;
    while ( v9 != -8 )
    {
      v44 = v41 + 1;
      v7 = (v5 - 1) & (v41 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v41 = v44;
    }
  }
  v8 = (__int64 *)(v6 + 16 * v5);
LABEL_3:
  v48 = v8[1];
  result = sub_1DD5D10(a3);
  v11 = *(_QWORD *)(a3 + 32);
  v47 = result;
  v46 = a1 + 56;
  if ( v11 == result )
    return result;
  do
  {
    v12 = *(_QWORD *)(v11 + 32);
    result = 5LL * (unsigned int)sub_1E163A0(v11);
    v13 = v12 + 8 * result;
    v14 = *(_QWORD *)(v11 + 32);
    if ( v13 == v14 )
      goto LABEL_14;
    do
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)v14 )
        {
          v15 = *(_DWORD *)(v14 + 8);
          if ( v15 < 0 )
            break;
        }
        v14 += 40;
        if ( v13 == v14 )
          goto LABEL_14;
      }
      v16 = *(_DWORD *)(a1 + 80);
      if ( !v16 )
      {
        ++*(_QWORD *)(a1 + 56);
        goto LABEL_30;
      }
      v17 = *(_QWORD *)(a1 + 64);
      v18 = (v16 - 1) & (37 * v15);
      v19 = (int *)(v17 + 8LL * v18);
      v20 = *v19;
      if ( v15 == *v19 )
      {
        v21 = v19[1];
        goto LABEL_12;
      }
      v24 = 1;
      v25 = 0;
      while ( 1 )
      {
        if ( v20 == -1 )
        {
          if ( !v25 )
            v25 = v19;
          v26 = *(_DWORD *)(a1 + 72);
          ++*(_QWORD *)(a1 + 56);
          v27 = v26 + 1;
          if ( 4 * v27 >= 3 * v16 )
          {
LABEL_30:
            sub_1BFDD60(v46, 2 * v16);
            v28 = *(_DWORD *)(a1 + 80);
            if ( v28 )
            {
              v29 = v28 - 1;
              v30 = *(_QWORD *)(a1 + 64);
              LODWORD(v31) = v29 & (37 * v15);
              v25 = (int *)(v30 + 8LL * (unsigned int)v31);
              v32 = *v25;
              v27 = *(_DWORD *)(a1 + 72) + 1;
              if ( v15 == *v25 )
                goto LABEL_26;
              v33 = 1;
              v34 = 0;
              while ( v32 != -1 )
              {
                if ( v32 == -2 && !v34 )
                  v34 = v25;
                v31 = v29 & (unsigned int)(v31 + v33);
                v25 = (int *)(v30 + 8 * v31);
                v32 = *v25;
                if ( v15 == *v25 )
                  goto LABEL_26;
                ++v33;
              }
LABEL_34:
              if ( v34 )
                v25 = v34;
              goto LABEL_26;
            }
          }
          else
          {
            if ( v16 - *(_DWORD *)(a1 + 76) - v27 > v16 >> 3 )
            {
LABEL_26:
              *(_DWORD *)(a1 + 72) = v27;
              if ( *v25 != -1 )
                --*(_DWORD *)(a1 + 76);
              *v25 = v15;
              v22 = 0;
              v25[1] = 0;
              v23 = -2;
              goto LABEL_13;
            }
            v45 = 37 * v15;
            sub_1BFDD60(v46, v16);
            v35 = *(_DWORD *)(a1 + 80);
            if ( v35 )
            {
              v36 = v35 - 1;
              v37 = *(_QWORD *)(a1 + 64);
              v38 = 1;
              v34 = 0;
              LODWORD(v39) = v36 & v45;
              v25 = (int *)(v37 + 8LL * (v36 & (unsigned int)v45));
              v40 = *v25;
              v27 = *(_DWORD *)(a1 + 72) + 1;
              if ( v15 == *v25 )
                goto LABEL_26;
              while ( v40 != -1 )
              {
                if ( v40 == -2 && !v34 )
                  v34 = v25;
                v39 = v36 & (unsigned int)(v39 + v38);
                v25 = (int *)(v37 + 8 * v39);
                v40 = *v25;
                if ( v15 == *v25 )
                  goto LABEL_26;
                ++v38;
              }
              goto LABEL_34;
            }
          }
          JUMPOUT(0x424C22);
        }
        if ( v20 != -2 || v25 )
          v19 = v25;
        v42 = v24 + 1;
        v18 = (v16 - 1) & (v24 + v18);
        v43 = (int *)(v17 + 8LL * v18);
        v20 = *v43;
        if ( v15 == *v43 )
          break;
        v24 = v42;
        v25 = v19;
        v19 = (int *)(v17 + 8LL * v18);
      }
      v21 = v43[1];
LABEL_12:
      v22 = 8LL * (v21 >> 6);
      v23 = ~(1LL << v21);
LABEL_13:
      v14 += 40;
      result = *(_QWORD *)(v48 + 48) + v22;
      *(_QWORD *)result &= v23;
    }
    while ( v13 != v14 );
LABEL_14:
    if ( (*(_BYTE *)v11 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v11 + 46) & 8) != 0 )
        v11 = *(_QWORD *)(v11 + 8);
    }
    v11 = *(_QWORD *)(v11 + 8);
  }
  while ( v47 != v11 );
  return result;
}

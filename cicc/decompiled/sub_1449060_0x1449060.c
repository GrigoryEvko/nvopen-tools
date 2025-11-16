// Function: sub_1449060
// Address: 0x1449060
//
void __fastcall sub_1449060(__int64 a1, __int64 *a2, _QWORD *a3)
{
  _QWORD *v5; // r12
  __int64 i; // rbx
  __int64 v7; // rsi
  __int64 v8; // rdx
  int v9; // r8d
  unsigned int v10; // r15d
  unsigned int v11; // edi
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r15
  _QWORD *v15; // rax
  __int64 *v16; // r12
  __int64 *k; // rbx
  __int64 v18; // rsi
  __int64 v19; // r10
  int j; // r11d
  _QWORD *v21; // r10
  int v22; // r11d
  int v23; // ecx
  int v24; // ecx
  int v25; // r9d
  int v26; // r9d
  __int64 v27; // r10
  __int64 v28; // rdx
  __int64 v29; // r8
  int v30; // edi
  _QWORD *v31; // rsi
  int v32; // edi
  int v33; // edi
  __int64 v34; // r8
  int v35; // edx
  __int64 v36; // r15
  _QWORD *v37; // r9
  __int64 v38; // rsi
  int v39; // r10d
  __int64 v40; // r11
  _QWORD *v41; // r11
  int v42; // [rsp+4h] [rbp-3Ch]
  unsigned int v43; // [rsp+8h] [rbp-38h]
  _QWORD *v44; // [rsp+8h] [rbp-38h]

  v5 = a3;
  for ( i = *a2; v5[4] == i; v5 = (_QWORD *)v5[1] )
    ;
  v7 = *(unsigned int *)(a1 + 64);
  v8 = *(_QWORD *)(a1 + 48);
  if ( !(_DWORD)v7 )
  {
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_24;
  }
  v9 = v7 - 1;
  v10 = ((unsigned int)i >> 9) ^ ((unsigned int)i >> 4);
  v11 = (v7 - 1) & v10;
  v12 = (_QWORD *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( i == *v12 )
  {
    if ( v12 != (_QWORD *)(v8 + 16 * v7) )
      goto LABEL_6;
    goto LABEL_22;
  }
  v43 = v9 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
  v19 = *v12;
  for ( j = 1; ; j = v42 )
  {
    if ( v19 == -8 )
    {
      v10 = ((unsigned int)i >> 4) ^ ((unsigned int)i >> 9);
      v11 = v10 & v9;
      v21 = (_QWORD *)(v8 + 16LL * (v10 & v9));
      v13 = *v21;
      if ( i == *v21 )
      {
        v12 = (_QWORD *)(v8 + 16LL * (v10 & v9));
LABEL_22:
        v12[1] = v5;
        v14 = (__int64)v5;
        goto LABEL_7;
      }
LABEL_13:
      v22 = 1;
      v12 = 0;
      while ( v13 != -8 )
      {
        if ( v12 || v13 != -16 )
          v21 = v12;
        v11 = v9 & (v22 + v11);
        v12 = (_QWORD *)(v8 + 16LL * v11);
        v13 = *v12;
        if ( i == *v12 )
          goto LABEL_22;
        ++v22;
        v44 = v21;
        v21 = (_QWORD *)(v8 + 16LL * v11);
        v12 = v44;
      }
      v23 = *(_DWORD *)(a1 + 56);
      if ( !v12 )
        v12 = v21;
      ++*(_QWORD *)(a1 + 40);
      v24 = v23 + 1;
      if ( 4 * v24 < (unsigned int)(3 * v7) )
      {
        if ( (int)v7 - *(_DWORD *)(a1 + 60) - v24 > (unsigned int)v7 >> 3 )
        {
LABEL_19:
          *(_DWORD *)(a1 + 56) = v24;
          if ( *v12 != -8 )
            --*(_DWORD *)(a1 + 60);
          *v12 = i;
          v12[1] = 0;
          goto LABEL_22;
        }
        sub_1448190(a1 + 40, v7);
        v32 = *(_DWORD *)(a1 + 64);
        if ( v32 )
        {
          v33 = v32 - 1;
          v34 = *(_QWORD *)(a1 + 48);
          v35 = 1;
          LODWORD(v36) = v33 & v10;
          v37 = 0;
          v24 = *(_DWORD *)(a1 + 56) + 1;
          v12 = (_QWORD *)(v34 + 16LL * (unsigned int)v36);
          v38 = *v12;
          if ( i != *v12 )
          {
            while ( v38 != -8 )
            {
              if ( v38 == -16 && !v37 )
                v37 = v12;
              v36 = v33 & (unsigned int)(v36 + v35);
              v12 = (_QWORD *)(v34 + 16 * v36);
              v38 = *v12;
              if ( i == *v12 )
                goto LABEL_19;
              ++v35;
            }
            if ( v37 )
              v12 = v37;
          }
          goto LABEL_19;
        }
LABEL_58:
        ++*(_DWORD *)(a1 + 56);
        BUG();
      }
LABEL_24:
      sub_1448190(a1 + 40, 2 * v7);
      v25 = *(_DWORD *)(a1 + 64);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(a1 + 48);
        LODWORD(v28) = v26 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
        v24 = *(_DWORD *)(a1 + 56) + 1;
        v12 = (_QWORD *)(v27 + 16LL * (unsigned int)v28);
        v29 = *v12;
        if ( i != *v12 )
        {
          v30 = 1;
          v31 = 0;
          while ( v29 != -8 )
          {
            if ( !v31 && v29 == -16 )
              v31 = v12;
            v28 = v26 & (unsigned int)(v28 + v30);
            v12 = (_QWORD *)(v27 + 16 * v28);
            v29 = *v12;
            if ( i == *v12 )
              goto LABEL_19;
            ++v30;
          }
          if ( v31 )
            v12 = v31;
        }
        goto LABEL_19;
      }
      goto LABEL_58;
    }
    v39 = j + 1;
    v40 = v9 & (v43 + j);
    v42 = v39;
    v43 = v40;
    v41 = (_QWORD *)(v8 + 16 * v40);
    v19 = *v41;
    if ( i == *v41 )
      break;
  }
  if ( v41 == (_QWORD *)(v8 + 16LL * (unsigned int)v7) )
  {
    v21 = (_QWORD *)(v8 + 16LL * (v9 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4))));
    goto LABEL_13;
  }
  v12 = v41;
LABEL_6:
  v14 = v12[1];
  v15 = (_QWORD *)sub_1443F10(a1, v14);
  sub_1448590(v5, v15, 0);
LABEL_7:
  v16 = (__int64 *)a2[4];
  for ( k = (__int64 *)a2[3]; v16 != k; ++k )
  {
    v18 = *k;
    sub_1449060(a1, v18, v14);
  }
}

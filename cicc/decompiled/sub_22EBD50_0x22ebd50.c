// Function: sub_22EBD50
// Address: 0x22ebd50
//
__int64 __fastcall sub_22EBD50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v12; // rdx
  const char *v13; // r10
  __int64 v14; // r11
  __int64 v15; // r13
  unsigned int v16; // r8d
  __int64 v17; // r9
  __int64 *v18; // rsi
  __int64 v19; // rcx
  _QWORD *v20; // rsi
  __int64 *v22; // rax
  int v23; // ecx
  int v24; // esi
  int v25; // r8d
  int v26; // r8d
  __int64 v27; // rcx
  unsigned int v28; // r9d
  __int64 *v29; // rdi
  __int64 v30; // r11
  int v31; // r8d
  int v32; // r8d
  __int64 v33; // rcx
  __int64 v34; // r9
  __int64 v35; // r11
  int v36; // [rsp+Ch] [rbp-64h]
  const char *v37; // [rsp+10h] [rbp-60h]
  int v38; // [rsp+10h] [rbp-60h]
  int v39; // [rsp+10h] [rbp-60h]
  __int64 v40; // [rsp+18h] [rbp-58h]
  unsigned int v41; // [rsp+20h] [rbp-50h]
  const char *v42; // [rsp+20h] [rbp-50h]
  __int64 v43; // [rsp+28h] [rbp-48h]

  *(_QWORD *)(a1 + 152) = a2;
  v13 = sub_BD5D20(a2);
  v14 = v12;
  v15 = *(_QWORD *)(a1 + 16) + 32LL * *(unsigned int *)(a1 + 24) - 32;
  v16 = *(_DWORD *)(v15 + 24);
  if ( !v16 )
  {
    ++*(_QWORD *)v15;
    goto LABEL_15;
  }
  v17 = *(_QWORD *)(v15 + 8);
  v41 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v18 = (__int64 *)(v17 + 72LL * v41);
  v19 = *v18;
  if ( *v18 == a2 )
    goto LABEL_3;
  v36 = 1;
  v22 = 0;
  v37 = v13;
  v40 = v12;
  while ( 1 )
  {
    if ( v19 == -4096 )
    {
      v23 = *(_DWORD *)(v15 + 16);
      if ( !v22 )
        v22 = v18;
      v14 = v12;
      ++*(_QWORD *)v15;
      v24 = v23 + 1;
      if ( 4 * (v23 + 1) < 3 * v16 )
      {
        if ( v16 - *(_DWORD *)(v15 + 20) - v24 > v16 >> 3 )
        {
LABEL_11:
          *(_DWORD *)(v15 + 16) = v24;
          if ( *v22 != -4096 )
            --*(_DWORD *)(v15 + 20);
          *v22 = a2;
          v20 = v22 + 1;
          *(_OWORD *)(v22 + 1) = 0;
          *(_OWORD *)(v22 + 3) = 0;
          *(_OWORD *)(v22 + 5) = 0;
          *(_OWORD *)(v22 + 7) = 0;
          return sub_3142480(a1, v20, v13, v14, a3, a4, a5, a6, a7, a8, *(_QWORD *)(a1 + 152));
        }
        sub_22EBA60(v15, v16);
        v31 = *(_DWORD *)(v15 + 24);
        if ( v31 )
        {
          v32 = v31 - 1;
          v33 = *(_QWORD *)(v15 + 8);
          v13 = v37;
          v14 = v40;
          v24 = *(_DWORD *)(v15 + 16) + 1;
          v34 = v32 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v22 = (__int64 *)(v33 + 72 * v34);
          if ( *v22 == a2 )
            goto LABEL_11;
          v39 = 1;
          v29 = 0;
          v42 = v13;
          v43 = v40;
          v35 = *v22;
          while ( v35 != -4096 )
          {
            if ( v35 == -8192 && !v29 )
              v29 = v22;
            LODWORD(v34) = v32 & (v39 + v34);
            v22 = (__int64 *)(v33 + 72LL * (unsigned int)v34);
            v35 = *v22;
            if ( *v22 == a2 )
              goto LABEL_30;
            ++v39;
          }
          goto LABEL_19;
        }
        goto LABEL_43;
      }
LABEL_15:
      v42 = v13;
      v43 = v14;
      sub_22EBA60(v15, 2 * v16);
      v25 = *(_DWORD *)(v15 + 24);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(v15 + 8);
        v13 = v42;
        v14 = v43;
        v28 = v26 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v24 = *(_DWORD *)(v15 + 16) + 1;
        v22 = (__int64 *)(v27 + 72LL * v28);
        if ( *v22 == a2 )
          goto LABEL_11;
        v38 = 1;
        v29 = 0;
        v30 = *v22;
        while ( v30 != -4096 )
        {
          if ( !v29 && v30 == -8192 )
            v29 = v22;
          v28 = v26 & (v38 + v28);
          v22 = (__int64 *)(v27 + 72LL * v28);
          v30 = *v22;
          if ( *v22 == a2 )
          {
LABEL_30:
            v13 = v42;
            v14 = v43;
            goto LABEL_11;
          }
          ++v38;
        }
LABEL_19:
        v13 = v42;
        v14 = v43;
        if ( v29 )
          v22 = v29;
        goto LABEL_11;
      }
LABEL_43:
      ++*(_DWORD *)(v15 + 16);
      BUG();
    }
    if ( v19 == -8192 && !v22 )
      v22 = v18;
    v41 = (v16 - 1) & (v41 + v36);
    v18 = (__int64 *)(v17 + 72LL * v41);
    v19 = *v18;
    if ( *v18 == a2 )
      break;
    ++v36;
  }
  v14 = v12;
LABEL_3:
  v20 = v18 + 1;
  return sub_3142480(a1, v20, v13, v14, a3, a4, a5, a6, a7, a8, *(_QWORD *)(a1 + 152));
}

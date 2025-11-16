// Function: sub_34E1EC0
// Address: 0x34e1ec0
//
__int64 __fastcall sub_34E1EC0(__int64 a1, __int64 *a2, unsigned __int8 *a3, size_t a4, unsigned __int8 *a5, size_t a6)
{
  const char *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r13
  char *v12; // r10
  __int64 v13; // r11
  __int64 v14; // rbx
  unsigned int v15; // r8d
  __int64 v16; // r9
  __int64 *v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 *v21; // rax
  int v22; // ecx
  int v23; // esi
  int v24; // r8d
  int v25; // r8d
  __int64 v26; // rcx
  unsigned int v27; // r9d
  __int64 *v28; // rdi
  __int64 v29; // r11
  int v30; // r8d
  int v31; // r8d
  __int64 v32; // rcx
  __int64 v33; // r9
  __int64 v34; // r11
  int v35; // [rsp+Ch] [rbp-64h]
  char *v36; // [rsp+10h] [rbp-60h]
  int v37; // [rsp+10h] [rbp-60h]
  int v38; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+18h] [rbp-58h]
  unsigned int v40; // [rsp+20h] [rbp-50h]
  char *v41; // [rsp+20h] [rbp-50h]
  __int64 v42; // [rsp+28h] [rbp-48h]

  *(_QWORD *)(a1 + 152) = a2;
  v9 = sub_2E791E0(a2);
  v11 = *a2;
  v12 = (char *)v9;
  v13 = v10;
  v14 = *(_QWORD *)(a1 + 16) + 32LL * *(unsigned int *)(a1 + 24) - 32;
  v15 = *(_DWORD *)(v14 + 24);
  if ( !v15 )
  {
    ++*(_QWORD *)v14;
    goto LABEL_15;
  }
  v16 = *(_QWORD *)(v14 + 8);
  v40 = (v15 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v17 = (__int64 *)(v16 + 72LL * v40);
  v18 = *v17;
  if ( v11 == *v17 )
    goto LABEL_3;
  v35 = 1;
  v21 = 0;
  v36 = v12;
  v39 = v10;
  while ( 1 )
  {
    if ( v18 == -4096 )
    {
      v22 = *(_DWORD *)(v14 + 16);
      if ( !v21 )
        v21 = v17;
      v13 = v10;
      ++*(_QWORD *)v14;
      v23 = v22 + 1;
      if ( 4 * (v22 + 1) < 3 * v15 )
      {
        if ( v15 - *(_DWORD *)(v14 + 20) - v23 > v15 >> 3 )
        {
LABEL_11:
          *(_DWORD *)(v14 + 16) = v23;
          if ( *v21 != -4096 )
            --*(_DWORD *)(v14 + 20);
          *v21 = v11;
          v19 = (__int64)(v21 + 1);
          *(_OWORD *)(v21 + 1) = 0;
          *(_OWORD *)(v21 + 3) = 0;
          *(_OWORD *)(v21 + 5) = 0;
          *(_OWORD *)(v21 + 7) = 0;
          return sub_3142480(a1, v19, v12, v13, a3, a4, a5, a6, (unsigned __int8 *)"MachineFunction", 0xFu, v11);
        }
        sub_22EBA60(v14, v15);
        v30 = *(_DWORD *)(v14 + 24);
        if ( v30 )
        {
          v31 = v30 - 1;
          v32 = *(_QWORD *)(v14 + 8);
          v12 = v36;
          v13 = v39;
          v23 = *(_DWORD *)(v14 + 16) + 1;
          v33 = v31 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v21 = (__int64 *)(v32 + 72 * v33);
          if ( v11 == *v21 )
            goto LABEL_11;
          v38 = 1;
          v28 = 0;
          v41 = v12;
          v42 = v39;
          v34 = *v21;
          while ( v34 != -4096 )
          {
            if ( v34 == -8192 && !v28 )
              v28 = v21;
            LODWORD(v33) = v31 & (v38 + v33);
            v21 = (__int64 *)(v32 + 72LL * (unsigned int)v33);
            v34 = *v21;
            if ( v11 == *v21 )
              goto LABEL_30;
            ++v38;
          }
          goto LABEL_19;
        }
        goto LABEL_43;
      }
LABEL_15:
      v41 = v12;
      v42 = v13;
      sub_22EBA60(v14, 2 * v15);
      v24 = *(_DWORD *)(v14 + 24);
      if ( v24 )
      {
        v25 = v24 - 1;
        v26 = *(_QWORD *)(v14 + 8);
        v12 = v41;
        v13 = v42;
        v27 = v25 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v23 = *(_DWORD *)(v14 + 16) + 1;
        v21 = (__int64 *)(v26 + 72LL * v27);
        if ( v11 == *v21 )
          goto LABEL_11;
        v37 = 1;
        v28 = 0;
        v29 = *v21;
        while ( v29 != -4096 )
        {
          if ( !v28 && v29 == -8192 )
            v28 = v21;
          v27 = v25 & (v37 + v27);
          v21 = (__int64 *)(v26 + 72LL * v27);
          v29 = *v21;
          if ( v11 == *v21 )
          {
LABEL_30:
            v12 = v41;
            v13 = v42;
            goto LABEL_11;
          }
          ++v37;
        }
LABEL_19:
        v12 = v41;
        v13 = v42;
        if ( v28 )
          v21 = v28;
        goto LABEL_11;
      }
LABEL_43:
      ++*(_DWORD *)(v14 + 16);
      BUG();
    }
    if ( v18 == -8192 && !v21 )
      v21 = v17;
    v40 = (v15 - 1) & (v35 + v40);
    v17 = (__int64 *)(v16 + 72LL * v40);
    v18 = *v17;
    if ( v11 == *v17 )
      break;
    ++v35;
  }
  v13 = v10;
LABEL_3:
  v19 = (__int64)(v17 + 1);
  return sub_3142480(a1, v19, v12, v13, a3, a4, a5, a6, (unsigned __int8 *)"MachineFunction", 0xFu, v11);
}

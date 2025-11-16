// Function: sub_1E0E7D0
// Address: 0x1e0e7d0
//
void __fastcall sub_1E0E7D0(__int64 a1, __int64 a2, const void *a3, __int64 a4)
{
  __int64 v4; // r9
  unsigned int v9; // esi
  __int64 *v10; // r8
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 *v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  signed __int64 v17; // r13
  __int64 v18; // r12
  int v19; // r11d
  __int64 *v20; // r10
  int v21; // ecx
  int v22; // ecx
  int v23; // eax
  int v24; // edx
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 v27; // rsi
  int v28; // edx
  int v29; // edx
  __int64 v30; // rdi
  unsigned int v31; // eax
  __int64 v32; // rsi
  unsigned int v33; // [rsp+8h] [rbp-38h]

  v4 = a1 + 432;
  v9 = *(_DWORD *)(a1 + 456);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 432);
    goto LABEL_19;
  }
  LODWORD(v10) = v9 - 1;
  v11 = *(_QWORD *)(a1 + 440);
  v12 = (v9 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v13 = (__int64 *)(v11 + 40LL * v12);
  v14 = *v13;
  if ( *v13 == a2 )
    goto LABEL_3;
  v19 = 1;
  v20 = 0;
  while ( 1 )
  {
    if ( v14 == -8 )
    {
      v21 = *(_DWORD *)(a1 + 448);
      if ( v20 )
        v13 = v20;
      ++*(_QWORD *)(a1 + 432);
      v22 = v21 + 1;
      if ( 4 * v22 < 3 * v9 )
      {
        if ( v9 - *(_DWORD *)(a1 + 452) - v22 > v9 >> 3 )
        {
LABEL_15:
          *(_DWORD *)(a1 + 448) = v22;
          if ( *v13 != -8 )
            --*(_DWORD *)(a1 + 452);
          *v13 = a2;
          v15 = 0;
          v13[1] = (__int64)(v13 + 3);
          v13[2] = 0x400000000LL;
          v16 = 4;
          goto LABEL_4;
        }
        v33 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
        sub_1E0E4E0(v4, v9);
        v28 = *(_DWORD *)(a1 + 456);
        if ( v28 )
        {
          v29 = v28 - 1;
          LODWORD(v4) = 1;
          v10 = 0;
          v30 = *(_QWORD *)(a1 + 440);
          v31 = v29 & v33;
          v13 = (__int64 *)(v30 + 40LL * (v29 & v33));
          v32 = *v13;
          v22 = *(_DWORD *)(a1 + 448) + 1;
          if ( *v13 == a2 )
            goto LABEL_15;
          while ( v32 != -8 )
          {
            if ( v32 == -16 && !v10 )
              v10 = v13;
            v31 = v29 & (v4 + v31);
            v13 = (__int64 *)(v30 + 40LL * v31);
            v32 = *v13;
            if ( *v13 == a2 )
              goto LABEL_15;
            LODWORD(v4) = v4 + 1;
          }
          goto LABEL_23;
        }
        goto LABEL_45;
      }
LABEL_19:
      sub_1E0E4E0(v4, 2 * v9);
      v23 = *(_DWORD *)(a1 + 456);
      if ( v23 )
      {
        v24 = v23 - 1;
        v25 = *(_QWORD *)(a1 + 440);
        v26 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v13 = (__int64 *)(v25 + 40LL * v26);
        v27 = *v13;
        v22 = *(_DWORD *)(a1 + 448) + 1;
        if ( *v13 == a2 )
          goto LABEL_15;
        LODWORD(v4) = 1;
        v10 = 0;
        while ( v27 != -8 )
        {
          if ( !v10 && v27 == -16 )
            v10 = v13;
          v26 = v24 & (v4 + v26);
          v13 = (__int64 *)(v25 + 40LL * v26);
          v27 = *v13;
          if ( *v13 == a2 )
            goto LABEL_15;
          LODWORD(v4) = v4 + 1;
        }
LABEL_23:
        if ( v10 )
          v13 = v10;
        goto LABEL_15;
      }
LABEL_45:
      ++*(_DWORD *)(a1 + 448);
      BUG();
    }
    if ( v20 || v14 != -16 )
      v13 = v20;
    v12 = (unsigned int)v10 & (v19 + v12);
    v14 = *(_QWORD *)(v11 + 40LL * v12);
    if ( v14 == a2 )
      break;
    ++v19;
    v20 = v13;
    v13 = (__int64 *)(v11 + 40LL * v12);
  }
  v13 = (__int64 *)(v11 + 40LL * v12);
LABEL_3:
  v15 = *((unsigned int *)v13 + 4);
  v16 = *((unsigned int *)v13 + 5) - v15;
LABEL_4:
  v17 = 4 * a4;
  v18 = v17 >> 2;
  if ( v17 >> 2 > v16 )
  {
    sub_16CD150((__int64)(v13 + 1), v13 + 3, v18 + v15, 4, (int)v10, v4);
    v15 = *((unsigned int *)v13 + 4);
  }
  if ( v17 )
  {
    memcpy((void *)(v13[1] + 4 * v15), a3, v17);
    LODWORD(v15) = *((_DWORD *)v13 + 4);
  }
  *((_DWORD *)v13 + 4) = v15 + v18;
}

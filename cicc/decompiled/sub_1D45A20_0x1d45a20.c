// Function: sub_1D45A20
// Address: 0x1d45a20
//
__int64 __fastcall sub_1D45A20(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4)
{
  __int64 v4; // r10
  unsigned int v9; // esi
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // edi
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  int v17; // r11d
  __int64 *v18; // rcx
  int v19; // eax
  __int64 v20; // r15
  __int128 v21; // rdi
  __int64 v22; // rax
  __int64 *v23; // rcx
  unsigned __int8 *v24; // rsi
  int v25; // eax
  int v26; // esi
  __int64 v27; // rdi
  unsigned int v28; // eax
  int v29; // r10d
  int v30; // eax
  int v31; // eax
  __int64 v32; // rdi
  unsigned int v33; // r15d
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 *v36; // [rsp+8h] [rbp-48h]
  __int64 *v37; // [rsp+8h] [rbp-48h]
  __int64 *v38; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v39; // [rsp+18h] [rbp-38h] BYREF

  v4 = a1 + 872;
  v9 = *(_DWORD *)(a1 + 896);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 872);
    goto LABEL_20;
  }
  v10 = *(_QWORD *)(a1 + 880);
  v11 = v9 - 1;
  v12 = v11 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__int64 *)(v10 + 16LL * v12);
  v14 = *v13;
  if ( *v13 != a2 )
  {
    v17 = 1;
    v18 = 0;
    while ( v14 != -8 )
    {
      if ( !v18 && v14 == -16 )
        v18 = v13;
      v12 = v11 & (v17 + v12);
      v13 = (__int64 *)(v10 + 16LL * v12);
      v14 = *v13;
      if ( *v13 == a2 )
        goto LABEL_3;
      ++v17;
    }
    if ( !v18 )
      v18 = v13;
    v19 = *(_DWORD *)(a1 + 888);
    ++*(_QWORD *)(a1 + 872);
    v15 = (unsigned int)(v19 + 1);
    if ( 4 * (int)v15 < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 892) - (unsigned int)v15 > v9 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 888) = v15;
        if ( *v18 != -8 )
          --*(_DWORD *)(a1 + 892);
        *v18 = a2;
        v18[1] = 0;
        goto LABEL_14;
      }
      sub_1D45860(v4, v9);
      v30 = *(_DWORD *)(a1 + 896);
      if ( v30 )
      {
        v31 = v30 - 1;
        v11 = 1;
        v10 = 0;
        v32 = *(_QWORD *)(a1 + 880);
        v33 = v31 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v15 = (unsigned int)(*(_DWORD *)(a1 + 888) + 1);
        v18 = (__int64 *)(v32 + 16LL * v33);
        v34 = *v18;
        if ( *v18 != a2 )
        {
          while ( v34 != -8 )
          {
            if ( v34 == -16 && !v10 )
              v10 = (__int64)v18;
            v33 = v31 & (v11 + v33);
            v18 = (__int64 *)(v32 + 16LL * v33);
            v34 = *v18;
            if ( *v18 == a2 )
              goto LABEL_11;
            v11 = (unsigned int)(v11 + 1);
          }
          if ( v10 )
            v18 = (__int64 *)v10;
        }
        goto LABEL_11;
      }
LABEL_50:
      ++*(_DWORD *)(a1 + 888);
      BUG();
    }
LABEL_20:
    sub_1D45860(v4, 2 * v9);
    v25 = *(_DWORD *)(a1 + 896);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 880);
      v28 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = (unsigned int)(*(_DWORD *)(a1 + 888) + 1);
      v18 = (__int64 *)(v27 + 16LL * v28);
      v10 = *v18;
      if ( *v18 != a2 )
      {
        v29 = 1;
        v11 = 0;
        while ( v10 != -8 )
        {
          if ( !v11 && v10 == -16 )
            v11 = (__int64)v18;
          v28 = v26 & (v29 + v28);
          v18 = (__int64 *)(v27 + 16LL * v28);
          v10 = *v18;
          if ( *v18 == a2 )
            goto LABEL_11;
          ++v29;
        }
        if ( v11 )
          v18 = (__int64 *)v11;
      }
      goto LABEL_11;
    }
    goto LABEL_50;
  }
LABEL_3:
  v15 = v13[1];
  if ( v15 )
    return v13[1];
  v18 = v13;
LABEL_14:
  v20 = *(_QWORD *)(a1 + 208);
  if ( v20 )
  {
    *(_QWORD *)(a1 + 208) = *(_QWORD *)v20;
  }
  else
  {
    v38 = v18;
    v35 = sub_145CBF0((__int64 *)(a1 + 216), 112, 8);
    v18 = v38;
    v20 = v35;
  }
  *((_QWORD *)&v21 + 1) = a4;
  v36 = v18;
  *(_QWORD *)&v21 = a3;
  v22 = sub_1D274F0(v21, v15, (__int64)v18, v10, v11);
  v39 = 0;
  v23 = v36;
  *(_QWORD *)v20 = 0;
  v24 = v39;
  *(_QWORD *)(v20 + 40) = v22;
  *(_QWORD *)(v20 + 8) = 0;
  *(_QWORD *)(v20 + 16) = 0;
  *(_WORD *)(v20 + 24) = 41;
  *(_DWORD *)(v20 + 28) = -1;
  *(_QWORD *)(v20 + 32) = 0;
  *(_QWORD *)(v20 + 48) = 0;
  *(_QWORD *)(v20 + 56) = 0x100000000LL;
  *(_DWORD *)(v20 + 64) = 0;
  *(_QWORD *)(v20 + 72) = v24;
  if ( v24 )
  {
    sub_1623210((__int64)&v39, v24, v20 + 72);
    v23 = v36;
  }
  *(_WORD *)(v20 + 80) &= 0xF000u;
  *(_WORD *)(v20 + 26) = 0;
  *(_QWORD *)(v20 + 88) = a2;
  v23[1] = v20;
  v37 = v23;
  sub_1D172A0(a1, v20);
  return v37[1];
}

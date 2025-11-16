// Function: sub_34148C0
// Address: 0x34148c0
//
unsigned __int64 __fastcall sub_34148C0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // r10
  unsigned int v9; // esi
  __int64 v10; // r8
  int v11; // r11d
  _QWORD *v12; // rdx
  unsigned int v13; // edi
  _QWORD *v14; // rax
  __int64 v15; // rcx
  unsigned __int64 *v16; // r15
  __int64 v17; // r8
  int v19; // eax
  int v20; // ecx
  __m128i *v21; // rax
  unsigned __int64 v22; // r13
  __m128i *v23; // r8
  int v24; // edx
  int v25; // r14d
  unsigned __int8 *v26; // rsi
  int v27; // eax
  int v28; // esi
  __int64 v29; // rdi
  unsigned int v30; // eax
  __int64 v31; // r8
  int v32; // r10d
  _QWORD *v33; // r9
  int v34; // eax
  int v35; // eax
  int v36; // r9d
  _QWORD *v37; // r8
  __int64 v38; // rdi
  unsigned int v39; // r15d
  __int64 v40; // rsi
  __int64 v41; // rcx
  unsigned __int64 v42; // rax
  __int64 v43; // rax
  __m128i *v44; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v45; // [rsp+18h] [rbp-38h] BYREF

  v4 = a1 + 992;
  v9 = *(_DWORD *)(a1 + 1016);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 992);
    goto LABEL_26;
  }
  v10 = *(_QWORD *)(a1 + 1000);
  v11 = 1;
  v12 = 0;
  v13 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = (_QWORD *)(v10 + 16LL * v13);
  v15 = *v14;
  if ( *v14 != a2 )
  {
    while ( v15 != -4096 )
    {
      if ( !v12 && v15 == -8192 )
        v12 = v14;
      v13 = (v9 - 1) & (v11 + v13);
      v14 = (_QWORD *)(v10 + 16LL * v13);
      v15 = *v14;
      if ( *v14 == a2 )
        goto LABEL_3;
      ++v11;
    }
    if ( !v12 )
      v12 = v14;
    v19 = *(_DWORD *)(a1 + 1008);
    ++*(_QWORD *)(a1 + 992);
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 1012) - v20 > v9 >> 3 )
      {
LABEL_15:
        *(_DWORD *)(a1 + 1008) = v20;
        if ( *v12 != -4096 )
          --*(_DWORD *)(a1 + 1012);
        *v12 = a2;
        v16 = v12 + 1;
        v12[1] = 0;
LABEL_18:
        v21 = sub_33ED250(a1, a3, a4);
        v22 = *(_QWORD *)(a1 + 416);
        v23 = v21;
        v25 = v24;
        if ( v22 )
        {
          *(_QWORD *)(a1 + 416) = *(_QWORD *)v22;
        }
        else
        {
          v41 = *(_QWORD *)(a1 + 424);
          *(_QWORD *)(a1 + 504) += 120LL;
          v42 = (v41 + 7) & 0xFFFFFFFFFFFFFFF8LL;
          if ( *(_QWORD *)(a1 + 432) >= v42 + 120 && v41 )
          {
            *(_QWORD *)(a1 + 424) = v42 + 120;
            if ( !v42 )
            {
LABEL_23:
              *v16 = v22;
              sub_33CC420(a1, v22);
              return *v16;
            }
            v22 = (v41 + 7) & 0xFFFFFFFFFFFFFFF8LL;
          }
          else
          {
            v44 = v23;
            v43 = sub_9D1E70(a1 + 424, 120, 120, 3);
            v23 = v44;
            v22 = v43;
          }
        }
        v45 = 0;
        *(_QWORD *)v22 = 0;
        v26 = v45;
        *(_QWORD *)(v22 + 8) = 0;
        *(_QWORD *)(v22 + 16) = 0;
        *(_QWORD *)(v22 + 24) = 44;
        *(_WORD *)(v22 + 34) = -1;
        *(_DWORD *)(v22 + 36) = -1;
        *(_QWORD *)(v22 + 40) = 0;
        *(_QWORD *)(v22 + 48) = v23;
        *(_QWORD *)(v22 + 56) = 0;
        *(_DWORD *)(v22 + 64) = 0;
        *(_DWORD *)(v22 + 68) = v25;
        *(_DWORD *)(v22 + 72) = 0;
        *(_QWORD *)(v22 + 80) = v26;
        if ( v26 )
          sub_B976B0((__int64)&v45, v26, v22 + 80);
        *(_QWORD *)(v22 + 88) = 0xFFFFFFFFLL;
        *(_WORD *)(v22 + 32) = 0;
        *(_QWORD *)(v22 + 96) = a2;
        goto LABEL_23;
      }
      sub_34146E0(v4, v9);
      v34 = *(_DWORD *)(a1 + 1016);
      if ( v34 )
      {
        v35 = v34 - 1;
        v36 = 1;
        v37 = 0;
        v38 = *(_QWORD *)(a1 + 1000);
        v39 = v35 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v20 = *(_DWORD *)(a1 + 1008) + 1;
        v12 = (_QWORD *)(v38 + 16LL * v39);
        v40 = *v12;
        if ( *v12 != a2 )
        {
          while ( v40 != -4096 )
          {
            if ( !v37 && v40 == -8192 )
              v37 = v12;
            v39 = v35 & (v36 + v39);
            v12 = (_QWORD *)(v38 + 16LL * v39);
            v40 = *v12;
            if ( *v12 == a2 )
              goto LABEL_15;
            ++v36;
          }
          if ( v37 )
            v12 = v37;
        }
        goto LABEL_15;
      }
LABEL_54:
      ++*(_DWORD *)(a1 + 1008);
      BUG();
    }
LABEL_26:
    sub_34146E0(v4, 2 * v9);
    v27 = *(_DWORD *)(a1 + 1016);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 1000);
      v30 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v20 = *(_DWORD *)(a1 + 1008) + 1;
      v12 = (_QWORD *)(v29 + 16LL * v30);
      v31 = *v12;
      if ( *v12 != a2 )
      {
        v32 = 1;
        v33 = 0;
        while ( v31 != -4096 )
        {
          if ( !v33 && v31 == -8192 )
            v33 = v12;
          v30 = v28 & (v32 + v30);
          v12 = (_QWORD *)(v29 + 16LL * v30);
          v31 = *v12;
          if ( *v12 == a2 )
            goto LABEL_15;
          ++v32;
        }
        if ( v33 )
          v12 = v33;
      }
      goto LABEL_15;
    }
    goto LABEL_54;
  }
LABEL_3:
  v16 = v14 + 1;
  v17 = v14[1];
  if ( !v17 )
    goto LABEL_18;
  return v17;
}

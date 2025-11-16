// Function: sub_39CC510
// Address: 0x39cc510
//
__int64 __fastcall sub_39CC510(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 v7; // r12
  __int64 v8; // r8
  unsigned int v9; // edi
  __int64 *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rcx
  int v16; // r8d
  __int64 v17; // r9
  __int64 v18; // r14
  __int64 v19; // rsi
  unsigned int v20; // eax
  __int64 v21; // r8
  __int64 v22; // rax
  int v24; // eax
  int v25; // esi
  __int64 v26; // rdi
  unsigned int v27; // eax
  int v28; // ecx
  __int64 *v29; // rdx
  __int64 v30; // r8
  int v31; // r10d
  __int64 *v32; // r9
  __int64 v33; // r8
  __int64 v34; // rax
  int v35; // r10d
  int v36; // eax
  int v37; // eax
  int v38; // eax
  __int64 v39; // rdi
  __int64 *v40; // r8
  unsigned int v41; // r15d
  int v42; // r9d
  __int64 v43; // rsi
  __int64 *v44; // r11
  _BYTE v45[2]; // [rsp+1Ch] [rbp-34h] BYREF
  char v46; // [rsp+1Eh] [rbp-32h]

  v4 = sub_15AB1E0(*(_BYTE **)(a2 + 8));
  if ( !sub_39C7370(a1) || (v7 = a1 + 864, (unsigned __int8)sub_3989C80(*(_QWORD *)(a1 + 200))) )
  {
    v5 = *(_QWORD *)(a1 + 208);
    v6 = *(_DWORD *)(v5 + 320);
    v7 = v5 + 296;
    if ( !v6 )
    {
LABEL_14:
      ++*(_QWORD *)v7;
      goto LABEL_15;
    }
  }
  else
  {
    v6 = *(_DWORD *)(a1 + 888);
    if ( !v6 )
      goto LABEL_14;
  }
  v8 = *(_QWORD *)(v7 + 8);
  v9 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (__int64 *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( v4 == *v10 )
  {
    v12 = v10[1];
    goto LABEL_5;
  }
  v35 = 1;
  v29 = 0;
  while ( v11 != -8 )
  {
    if ( v29 || v11 != -16 )
      v10 = v29;
    v9 = (v6 - 1) & (v35 + v9);
    v44 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v44;
    if ( v4 == *v44 )
    {
      v12 = v44[1];
      goto LABEL_5;
    }
    ++v35;
    v29 = v10;
    v10 = (__int64 *)(v8 + 16LL * v9);
  }
  if ( !v29 )
    v29 = v10;
  v36 = *(_DWORD *)(v7 + 16);
  ++*(_QWORD *)v7;
  v28 = v36 + 1;
  if ( 4 * (v36 + 1) >= 3 * v6 )
  {
LABEL_15:
    sub_39A53F0(v7, 2 * v6);
    v24 = *(_DWORD *)(v7 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(v7 + 8);
      v27 = (v24 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v28 = *(_DWORD *)(v7 + 16) + 1;
      v29 = (__int64 *)(v26 + 16LL * v27);
      v30 = *v29;
      if ( v4 != *v29 )
      {
        v31 = 1;
        v32 = 0;
        while ( v30 != -8 )
        {
          if ( !v32 && v30 == -16 )
            v32 = v29;
          v27 = v25 & (v31 + v27);
          v29 = (__int64 *)(v26 + 16LL * v27);
          v30 = *v29;
          if ( v4 == *v29 )
            goto LABEL_32;
          ++v31;
        }
        if ( v32 )
          v29 = v32;
      }
      goto LABEL_32;
    }
LABEL_57:
    ++*(_DWORD *)(v7 + 16);
    BUG();
  }
  if ( v6 - *(_DWORD *)(v7 + 20) - v28 <= v6 >> 3 )
  {
    sub_39A53F0(v7, v6);
    v37 = *(_DWORD *)(v7 + 24);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(v7 + 8);
      v40 = 0;
      v41 = v38 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v42 = 1;
      v28 = *(_DWORD *)(v7 + 16) + 1;
      v29 = (__int64 *)(v39 + 16LL * v41);
      v43 = *v29;
      if ( v4 != *v29 )
      {
        while ( v43 != -8 )
        {
          if ( v43 == -16 && !v40 )
            v40 = v29;
          v41 = v38 & (v42 + v41);
          v29 = (__int64 *)(v39 + 16LL * v41);
          v43 = *v29;
          if ( v4 == *v29 )
            goto LABEL_32;
          ++v42;
        }
        if ( v40 )
          v29 = v40;
      }
      goto LABEL_32;
    }
    goto LABEL_57;
  }
LABEL_32:
  *(_DWORD *)(v7 + 16) = v28;
  if ( *v29 != -8 )
    --*(_DWORD *)(v7 + 20);
  *v29 = v4;
  v12 = 0;
  v29[1] = 0;
LABEL_5:
  v13 = sub_145CDC0(0x30u, (__int64 *)(a1 + 88));
  v14 = v13;
  if ( v13 )
  {
    *(_QWORD *)(v13 + 8) = 0;
    *(_QWORD *)v13 = v13 | 4;
    *(_QWORD *)(v13 + 16) = 0;
    *(_DWORD *)(v13 + 24) = -1;
    *(_WORD *)(v13 + 28) = 29;
    *(_BYTE *)(v13 + 30) = 0;
    *(_QWORD *)(v13 + 32) = 0;
    *(_QWORD *)(v13 + 40) = 0;
  }
  sub_39A3B20(a1, v13, 49, v12);
  sub_39CB2D0((__int64 *)a1, v14, a2 + 80, v15, v16, v17);
  v18 = *(_QWORD *)(a2 + 16);
  v19 = *(_QWORD *)(v18 - 8LL * *(unsigned int *)(v18 + 8));
  if ( *(_BYTE *)v19 != 15 )
    v19 = *(_QWORD *)(v19 - 8LL * *(unsigned int *)(v19 + 8));
  v20 = sub_39CC330(a1, v19);
  v46 = 0;
  sub_39A3560(a1, (__int64 *)(v14 + 8), 88, (__int64)v45, v20);
  v21 = *(unsigned int *)(v18 + 4);
  v46 = 0;
  sub_39A3560(a1, (__int64 *)(v14 + 8), 89, (__int64)v45, v21);
  v22 = *(_QWORD *)(v18 - 8LL * *(unsigned int *)(v18 + 8));
  if ( *(_BYTE *)v22 == 19 && *(_DWORD *)(v22 + 24) && (unsigned __int16)sub_398C0A0(*(_QWORD *)(a1 + 200)) > 3u )
  {
    v33 = 0;
    v34 = *(_QWORD *)(v18 - 8LL * *(unsigned int *)(v18 + 8));
    if ( *(_BYTE *)v34 == 19 )
      v33 = *(unsigned int *)(v34 + 24);
    v46 = 0;
    sub_39A3560(a1, (__int64 *)(v14 + 8), 8502, (__int64)v45, v33);
  }
  sub_398FCD0(*(_QWORD *)(a1 + 200), v4, v14);
  return v14;
}

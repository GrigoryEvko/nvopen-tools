// Function: sub_2451E00
// Address: 0x2451e00
//
__int64 __fastcall sub_2451E00(__int64 a1, __int64 a2)
{
  unsigned __int8 *v4; // rax
  unsigned int v5; // esi
  __int64 v6; // rdi
  unsigned __int8 *v7; // r13
  __int64 v8; // r8
  int v9; // r11d
  unsigned int v10; // r14d
  unsigned int v11; // ecx
  _QWORD *v12; // rdx
  unsigned __int8 **v13; // rax
  unsigned __int8 *v14; // r10
  __int64 result; // rax
  _QWORD *v16; // r13
  int v17; // ecx
  int v18; // ecx
  __int64 v19; // rdx
  bool v20; // cc
  _QWORD *v21; // rdx
  int v22; // eax
  int v23; // esi
  __int64 v24; // rdi
  unsigned int v25; // edx
  unsigned __int8 *v26; // r8
  int v27; // r10d
  unsigned __int8 **v28; // r9
  int v29; // eax
  int v30; // edx
  __int64 v31; // rdi
  int v32; // r9d
  unsigned __int8 **v33; // r8
  unsigned int v34; // r14d
  unsigned __int8 *v35; // rsi

  v4 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), a2);
  v5 = *(_DWORD *)(a1 + 176);
  v6 = a1 + 152;
  v7 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 152);
    goto LABEL_22;
  }
  v8 = *(_QWORD *)(a1 + 160);
  v9 = 1;
  v10 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
  v11 = (v5 - 1) & v10;
  v12 = (_QWORD *)(v8 + 56LL * v11);
  v13 = 0;
  v14 = (unsigned __int8 *)*v12;
  if ( v7 == (unsigned __int8 *)*v12 )
  {
LABEL_3:
    result = v12[5];
    v16 = v12 + 1;
    if ( result )
      return result;
    goto LABEL_18;
  }
  while ( v14 != (unsigned __int8 *)-4096LL )
  {
    if ( v14 == (unsigned __int8 *)-8192LL && !v13 )
      v13 = (unsigned __int8 **)v12;
    v11 = (v5 - 1) & (v9 + v11);
    v12 = (_QWORD *)(v8 + 56LL * v11);
    v14 = (unsigned __int8 *)*v12;
    if ( v7 == (unsigned __int8 *)*v12 )
      goto LABEL_3;
    ++v9;
  }
  v17 = *(_DWORD *)(a1 + 168);
  if ( !v13 )
    v13 = (unsigned __int8 **)v12;
  ++*(_QWORD *)(a1 + 152);
  v18 = v17 + 1;
  if ( 4 * v18 >= 3 * v5 )
  {
LABEL_22:
    sub_24507C0(v6, 2 * v5);
    v22 = *(_DWORD *)(a1 + 176);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 160);
      v25 = (v22 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v13 = (unsigned __int8 **)(v24 + 56LL * v25);
      v26 = *v13;
      v18 = *(_DWORD *)(a1 + 168) + 1;
      if ( v7 != *v13 )
      {
        v27 = 1;
        v28 = 0;
        while ( v26 != (unsigned __int8 *)-4096LL )
        {
          if ( !v28 && v26 == (unsigned __int8 *)-8192LL )
            v28 = v13;
          v25 = v23 & (v27 + v25);
          v13 = (unsigned __int8 **)(v24 + 56LL * v25);
          v26 = *v13;
          if ( v7 == *v13 )
            goto LABEL_15;
          ++v27;
        }
        if ( v28 )
          v13 = v28;
      }
      goto LABEL_15;
    }
    goto LABEL_45;
  }
  if ( v5 - *(_DWORD *)(a1 + 172) - v18 <= v5 >> 3 )
  {
    sub_24507C0(v6, v5);
    v29 = *(_DWORD *)(a1 + 176);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 160);
      v32 = 1;
      v33 = 0;
      v34 = (v29 - 1) & v10;
      v13 = (unsigned __int8 **)(v31 + 56LL * v34);
      v35 = *v13;
      v18 = *(_DWORD *)(a1 + 168) + 1;
      if ( v7 != *v13 )
      {
        while ( v35 != (unsigned __int8 *)-4096LL )
        {
          if ( v35 == (unsigned __int8 *)-8192LL && !v33 )
            v33 = v13;
          v34 = v30 & (v32 + v34);
          v13 = (unsigned __int8 **)(v31 + 56LL * v34);
          v35 = *v13;
          if ( v7 == *v13 )
            goto LABEL_15;
          ++v32;
        }
        if ( v33 )
          v13 = v33;
      }
      goto LABEL_15;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 168);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 168) = v18;
  if ( *v13 != (unsigned __int8 *)-4096LL )
    --*(_DWORD *)(a1 + 172);
  *v13 = v7;
  v16 = v13 + 1;
  *(_OWORD *)(v13 + 1) = 0;
  *(_OWORD *)(v13 + 3) = 0;
  *(_OWORD *)(v13 + 5) = 0;
LABEL_18:
  result = sub_2451540((_QWORD **)a1, a2, 2);
  v16[4] = result;
  v19 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v20 = *(_DWORD *)(v19 + 32) <= 0x40u;
  v21 = *(_QWORD **)(v19 + 24);
  if ( !v20 )
    v21 = (_QWORD *)*v21;
  *((_DWORD *)v16 + 10) = (v21 != 0) + (unsigned int)(((unsigned __int64)v21 - (v21 != 0)) >> 3);
  return result;
}

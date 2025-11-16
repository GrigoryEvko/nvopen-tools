// Function: sub_1B86660
// Address: 0x1b86660
//
__int64 __fastcall sub_1B86660(__int64 *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r9
  __int64 result; // rax
  int v11; // edx
  __int64 v12; // r12
  unsigned int v13; // esi
  __int64 v14; // r9
  unsigned int v15; // edi
  __int64 *v16; // rdx
  __int64 v17; // rcx
  int v18; // r10d
  int v19; // r14d
  __int64 *v20; // r11
  int v21; // edi
  int v22; // edi
  int v23; // edx
  int v24; // esi
  __int64 v25; // r9
  unsigned int v26; // ecx
  __int64 v27; // r8
  int v28; // r11d
  __int64 *v29; // r10
  int v30; // edx
  int v31; // ecx
  __int64 v32; // r8
  __int64 *v33; // r9
  unsigned int v34; // r13d
  int v35; // r10d
  __int64 v36; // rsi
  __int64 v37; // [rsp+8h] [rbp-38h]
  __int64 v38; // [rsp+8h] [rbp-38h]

  v4 = *a1;
  v5 = *(unsigned int *)(*a1 + 152);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(v4 + 136);
    v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v6 + 16 * v5) )
        return v8[1];
    }
    else
    {
      v11 = 1;
      while ( v9 != -8 )
      {
        v18 = v11 + 1;
        v7 = (v5 - 1) & (v11 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == a2 )
          goto LABEL_3;
        v11 = v18;
      }
    }
  }
  result = sub_146F1B0(*(_QWORD *)(v4 + 24), a2);
  v12 = *a1;
  v13 = *(_DWORD *)(v12 + 152);
  if ( !v13 )
  {
    ++*(_QWORD *)(v12 + 128);
    goto LABEL_22;
  }
  v14 = *(_QWORD *)(v12 + 136);
  v15 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v16 = (__int64 *)(v14 + 16LL * v15);
  v17 = *v16;
  if ( *v16 != a2 )
  {
    v19 = 1;
    v20 = 0;
    while ( v17 != -8 )
    {
      if ( !v20 && v17 == -16 )
        v20 = v16;
      v15 = (v13 - 1) & (v19 + v15);
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == a2 )
        goto LABEL_9;
      ++v19;
    }
    v21 = *(_DWORD *)(v12 + 144);
    if ( v20 )
      v16 = v20;
    ++*(_QWORD *)(v12 + 128);
    v22 = v21 + 1;
    if ( 4 * v22 < 3 * v13 )
    {
      if ( v13 - *(_DWORD *)(v12 + 148) - v22 > v13 >> 3 )
      {
LABEL_18:
        *(_DWORD *)(v12 + 144) = v22;
        if ( *v16 != -8 )
          --*(_DWORD *)(v12 + 148);
        *v16 = a2;
        v16[1] = 0;
        goto LABEL_9;
      }
      v38 = result;
      sub_1B864A0(v12 + 128, v13);
      v30 = *(_DWORD *)(v12 + 152);
      if ( v30 )
      {
        v31 = v30 - 1;
        v32 = *(_QWORD *)(v12 + 136);
        v33 = 0;
        v34 = (v30 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v35 = 1;
        v22 = *(_DWORD *)(v12 + 144) + 1;
        result = v38;
        v16 = (__int64 *)(v32 + 16LL * v34);
        v36 = *v16;
        if ( *v16 != a2 )
        {
          while ( v36 != -8 )
          {
            if ( v36 == -16 && !v33 )
              v33 = v16;
            v34 = v31 & (v35 + v34);
            v16 = (__int64 *)(v32 + 16LL * v34);
            v36 = *v16;
            if ( *v16 == a2 )
              goto LABEL_18;
            ++v35;
          }
          if ( v33 )
            v16 = v33;
        }
        goto LABEL_18;
      }
LABEL_50:
      ++*(_DWORD *)(v12 + 144);
      BUG();
    }
LABEL_22:
    v37 = result;
    sub_1B864A0(v12 + 128, 2 * v13);
    v23 = *(_DWORD *)(v12 + 152);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(v12 + 136);
      v26 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v22 = *(_DWORD *)(v12 + 144) + 1;
      result = v37;
      v16 = (__int64 *)(v25 + 16LL * v26);
      v27 = *v16;
      if ( *v16 != a2 )
      {
        v28 = 1;
        v29 = 0;
        while ( v27 != -8 )
        {
          if ( v27 == -16 && !v29 )
            v29 = v16;
          v26 = v24 & (v28 + v26);
          v16 = (__int64 *)(v25 + 16LL * v26);
          v27 = *v16;
          if ( *v16 == a2 )
            goto LABEL_18;
          ++v28;
        }
        if ( v29 )
          v16 = v29;
      }
      goto LABEL_18;
    }
    goto LABEL_50;
  }
LABEL_9:
  v16[1] = result;
  return result;
}

// Function: sub_AEBD40
// Address: 0xaebd40
//
__int64 __fastcall sub_AEBD40(__int64 *a1, __int64 a2)
{
  __int64 v4; // rcx
  int v5; // eax
  int v6; // esi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rdi
  __int64 result; // rax
  int v11; // eax
  _QWORD *v12; // rdi
  __int64 v13; // r12
  unsigned int v14; // esi
  __int64 v15; // r9
  int v16; // r14d
  _QWORD *v17; // rdi
  unsigned int v18; // r8d
  _QWORD *v19; // rdx
  __int64 v20; // rcx
  __int64 *v21; // rdx
  int v22; // edx
  int v23; // edx
  __int64 v24; // r9
  unsigned int v25; // esi
  int v26; // ecx
  __int64 v27; // r8
  int v28; // r8d
  int v29; // ecx
  int v30; // edx
  int v31; // esi
  __int64 v32; // r8
  _QWORD *v33; // r9
  unsigned int v34; // r13d
  int v35; // r10d
  __int64 v36; // rdx
  int v37; // r11d
  _QWORD *v38; // r10
  __int64 v39; // [rsp+8h] [rbp-28h]
  __int64 v40; // [rsp+8h] [rbp-28h]

  v4 = *(_QWORD *)(*a1 + 8);
  v5 = *(_DWORD *)(*a1 + 24);
  if ( !v5 )
    goto LABEL_7;
  v6 = v5 - 1;
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v4 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v11 = 1;
    while ( v9 != -4096 )
    {
      v28 = v11 + 1;
      v7 = v6 & (v11 + v7);
      v8 = (__int64 *)(v4 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v11 = v28;
    }
LABEL_7:
    v12 = (_QWORD *)(*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(a2 + 8) & 4) != 0 )
      v12 = (_QWORD *)*v12;
    result = sub_AF40E0(v12, 1, 1);
    v13 = *a1;
    v14 = *(_DWORD *)(v13 + 24);
    if ( v14 )
    {
      v15 = *(_QWORD *)(v13 + 8);
      v16 = 1;
      v17 = 0;
      v18 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = (_QWORD *)(v15 + 16LL * v18);
      v20 = *v19;
      if ( a2 == *v19 )
      {
LABEL_11:
        v21 = v19 + 1;
LABEL_12:
        *v21 = result;
        return result;
      }
      while ( v20 != -4096 )
      {
        if ( v20 == -8192 && !v17 )
          v17 = v19;
        v18 = (v14 - 1) & (v16 + v18);
        v19 = (_QWORD *)(v15 + 16LL * v18);
        v20 = *v19;
        if ( a2 == *v19 )
          goto LABEL_11;
        ++v16;
      }
      v29 = *(_DWORD *)(v13 + 16);
      if ( !v17 )
        v17 = v19;
      ++*(_QWORD *)v13;
      v26 = v29 + 1;
      if ( 4 * v26 < 3 * v14 )
      {
        if ( v14 - *(_DWORD *)(v13 + 20) - v26 > v14 >> 3 )
        {
LABEL_16:
          *(_DWORD *)(v13 + 16) = v26;
          if ( *v17 != -4096 )
            --*(_DWORD *)(v13 + 20);
          *v17 = a2;
          v21 = v17 + 1;
          v17[1] = 0;
          goto LABEL_12;
        }
        v40 = result;
        sub_AEBB60(v13, v14);
        v30 = *(_DWORD *)(v13 + 24);
        if ( v30 )
        {
          v31 = v30 - 1;
          v32 = *(_QWORD *)(v13 + 8);
          v33 = 0;
          v34 = (v30 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v35 = 1;
          v26 = *(_DWORD *)(v13 + 16) + 1;
          result = v40;
          v17 = (_QWORD *)(v32 + 16LL * v34);
          v36 = *v17;
          if ( a2 != *v17 )
          {
            while ( v36 != -4096 )
            {
              if ( !v33 && v36 == -8192 )
                v33 = v17;
              v34 = v31 & (v35 + v34);
              v17 = (_QWORD *)(v32 + 16LL * v34);
              v36 = *v17;
              if ( a2 == *v17 )
                goto LABEL_16;
              ++v35;
            }
            if ( v33 )
              v17 = v33;
          }
          goto LABEL_16;
        }
LABEL_52:
        ++*(_DWORD *)(v13 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)v13;
    }
    v39 = result;
    sub_AEBB60(v13, 2 * v14);
    v22 = *(_DWORD *)(v13 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v13 + 8);
      v25 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v26 = *(_DWORD *)(v13 + 16) + 1;
      result = v39;
      v17 = (_QWORD *)(v24 + 16LL * v25);
      v27 = *v17;
      if ( a2 != *v17 )
      {
        v37 = 1;
        v38 = 0;
        while ( v27 != -4096 )
        {
          if ( v27 == -8192 && !v38 )
            v38 = v17;
          v25 = v23 & (v37 + v25);
          v17 = (_QWORD *)(v24 + 16LL * v25);
          v27 = *v17;
          if ( a2 == *v17 )
            goto LABEL_16;
          ++v37;
        }
        if ( v38 )
          v17 = v38;
      }
      goto LABEL_16;
    }
    goto LABEL_52;
  }
LABEL_3:
  result = v8[1];
  if ( !result )
    goto LABEL_7;
  return result;
}

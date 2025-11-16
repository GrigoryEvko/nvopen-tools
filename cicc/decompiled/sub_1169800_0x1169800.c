// Function: sub_1169800
// Address: 0x1169800
//
__int64 __fastcall sub_1169800(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  char v6; // r8
  __int64 v7; // rdi
  int v8; // r10d
  unsigned int v9; // r9d
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 result; // rax
  __int64 v14; // rax
  char v15; // di
  unsigned int v16; // esi
  __int64 v17; // r8
  int v18; // r9d
  unsigned int v19; // esi
  _QWORD *v20; // rdx
  __int64 v21; // rcx
  __int64 *v22; // rdx
  __int64 v23; // rax
  int v24; // eax
  unsigned int v25; // ecx
  _QWORD *v26; // r10
  int v27; // edx
  unsigned int v28; // r8d
  int v29; // r11d
  __int64 v30; // rdi
  int v31; // esi
  unsigned int v32; // ecx
  __int64 v33; // rdx
  __int64 v34; // rdi
  int v35; // esi
  unsigned int v36; // ecx
  __int64 v37; // rdx
  int v38; // r9d
  _QWORD *v39; // r8
  int v40; // esi
  int v41; // r11d
  int v42; // esi
  int v43; // r9d
  __int64 v44; // [rsp+8h] [rbp-28h]
  __int64 v45; // [rsp+8h] [rbp-28h]

  v6 = *(_BYTE *)(a1 + 352) & 1;
  if ( v6 )
  {
    v7 = a1 + 360;
    v8 = 3;
  }
  else
  {
    v14 = *(unsigned int *)(a1 + 368);
    v7 = *(_QWORD *)(a1 + 360);
    if ( !(_DWORD)v14 )
      goto LABEL_18;
    v8 = v14 - 1;
  }
  v9 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
    goto LABEL_4;
  v24 = 1;
  while ( v11 != -4096 )
  {
    v41 = v24 + 1;
    v9 = v8 & (v24 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      goto LABEL_4;
    v24 = v41;
  }
  if ( v6 )
  {
    v23 = 64;
    goto LABEL_19;
  }
  v14 = *(unsigned int *)(a1 + 368);
LABEL_18:
  v23 = 16 * v14;
LABEL_19:
  v10 = (__int64 *)(v7 + v23);
LABEL_4:
  v12 = 64;
  if ( !v6 )
    v12 = 16LL * *(unsigned int *)(a1 + 368);
  if ( v10 != (__int64 *)(v7 + v12) )
    return v10[1];
  result = sub_116C930(a1, a2, a3);
  v15 = *(_BYTE *)(a1 + 352) & 1;
  if ( v15 )
  {
    v17 = a1 + 360;
    v18 = 3;
  }
  else
  {
    v16 = *(_DWORD *)(a1 + 368);
    v17 = *(_QWORD *)(a1 + 360);
    if ( !v16 )
    {
      v25 = *(_DWORD *)(a1 + 352);
      ++*(_QWORD *)(a1 + 344);
      v26 = 0;
      v27 = (v25 >> 1) + 1;
      goto LABEL_25;
    }
    v18 = v16 - 1;
  }
  v19 = v18 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v20 = (_QWORD *)(v17 + 16LL * v19);
  v21 = *v20;
  if ( *v20 != a2 )
  {
    v29 = 1;
    v26 = 0;
    while ( v21 != -4096 )
    {
      if ( v21 == -8192 && !v26 )
        v26 = v20;
      v19 = v18 & (v29 + v19);
      v20 = (_QWORD *)(v17 + 16LL * v19);
      v21 = *v20;
      if ( *v20 == a2 )
        goto LABEL_14;
      ++v29;
    }
    v25 = *(_DWORD *)(a1 + 352);
    v28 = 12;
    v16 = 4;
    if ( !v26 )
      v26 = v20;
    ++*(_QWORD *)(a1 + 344);
    v27 = (v25 >> 1) + 1;
    if ( v15 )
    {
LABEL_26:
      if ( 4 * v27 < v28 )
      {
        if ( v16 - *(_DWORD *)(a1 + 356) - v27 > v16 >> 3 )
        {
LABEL_28:
          *(_DWORD *)(a1 + 352) = (2 * (v25 >> 1) + 2) | v25 & 1;
          if ( *v26 != -4096 )
            --*(_DWORD *)(a1 + 356);
          *v26 = a2;
          v22 = v26 + 1;
          v26[1] = 0;
          goto LABEL_15;
        }
        v45 = result;
        sub_11693E0(a1 + 344, v16);
        result = v45;
        if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
        {
          v34 = a1 + 360;
          v35 = 3;
          goto LABEL_43;
        }
        v42 = *(_DWORD *)(a1 + 368);
        v34 = *(_QWORD *)(a1 + 360);
        if ( v42 )
        {
          v35 = v42 - 1;
LABEL_43:
          v36 = v35 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v26 = (_QWORD *)(v34 + 16LL * v36);
          v37 = *v26;
          if ( *v26 != a2 )
          {
            v38 = 1;
            v39 = 0;
            while ( v37 != -4096 )
            {
              if ( v37 == -8192 && !v39 )
                v39 = v26;
              v36 = v35 & (v38 + v36);
              v26 = (_QWORD *)(v34 + 16LL * v36);
              v37 = *v26;
              if ( *v26 == a2 )
                goto LABEL_40;
              ++v38;
            }
LABEL_46:
            if ( v39 )
              v26 = v39;
            goto LABEL_40;
          }
          goto LABEL_40;
        }
LABEL_71:
        *(_DWORD *)(a1 + 352) = (2 * (*(_DWORD *)(a1 + 352) >> 1) + 2) | *(_DWORD *)(a1 + 352) & 1;
        BUG();
      }
      v44 = result;
      sub_11693E0(a1 + 344, 2 * v16);
      result = v44;
      if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
      {
        v30 = a1 + 360;
        v31 = 3;
      }
      else
      {
        v40 = *(_DWORD *)(a1 + 368);
        v30 = *(_QWORD *)(a1 + 360);
        if ( !v40 )
          goto LABEL_71;
        v31 = v40 - 1;
      }
      v32 = v31 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v26 = (_QWORD *)(v30 + 16LL * v32);
      v33 = *v26;
      if ( *v26 != a2 )
      {
        v43 = 1;
        v39 = 0;
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v39 )
            v39 = v26;
          v32 = v31 & (v43 + v32);
          v26 = (_QWORD *)(v30 + 16LL * v32);
          v33 = *v26;
          if ( *v26 == a2 )
            goto LABEL_40;
          ++v43;
        }
        goto LABEL_46;
      }
LABEL_40:
      v25 = *(_DWORD *)(a1 + 352);
      goto LABEL_28;
    }
    v16 = *(_DWORD *)(a1 + 368);
LABEL_25:
    v28 = 3 * v16;
    goto LABEL_26;
  }
LABEL_14:
  v22 = v20 + 1;
LABEL_15:
  *v22 = result;
  return result;
}

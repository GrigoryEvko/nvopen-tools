// Function: sub_2B1EA40
// Address: 0x2b1ea40
//
__int64 __fastcall sub_2B1EA40(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        signed int a5,
        int a6,
        __int64 (__fastcall *a7)(__int64, _QWORD),
        __int64 a8)
{
  __int64 result; // rax
  int v12; // ebx
  __int64 v13; // r14
  unsigned __int64 v14; // rdx
  int v15; // r10d
  int *v16; // rdx
  __int64 v17; // r8
  _DWORD *v18; // rcx
  __int64 v19; // rdi
  _DWORD *v20; // r9
  __int64 v21; // rsi
  __int64 v22; // rdi
  int v23; // esi
  _DWORD *v24; // rdi
  __int64 v25; // rax
  int v26; // r11d
  _DWORD *v27; // r12
  __int64 v28; // rsi
  _DWORD *v29; // r9
  __int64 v30; // rcx
  __int64 v31; // rax
  int v32; // edi
  __int64 v33; // rax
  _DWORD *v34; // rcx
  _DWORD *v35; // r9
  __int64 v36; // rsi
  int v37; // edx
  unsigned int *v38; // r10
  unsigned int v39; // ecx
  int v40; // edx
  signed int v41; // edx
  _DWORD *v42; // rcx
  int j; // eax
  __int64 *v44; // rdx
  unsigned __int64 v45; // rax
  __int64 v46; // rsi
  unsigned __int8 *v47; // rdi
  __int64 v48; // rax
  _DWORD *v49; // rcx
  _DWORD *v50; // r9
  int i; // eax
  __int64 v52; // rsi
  int *v53; // [rsp+8h] [rbp-68h]
  _DWORD *v54; // [rsp+8h] [rbp-68h]
  __int64 v57; // [rsp+30h] [rbp-40h]

  result = a6;
  v57 = a6;
  if ( !a6 )
    return result;
  v12 = 0;
  v13 = 0;
  do
  {
    v14 = **(_QWORD **)a1;
    if ( (v14 & 1) != 0 )
      result = (((v14 >> 1) & ~(-1LL << (**(_QWORD **)a1 >> 58))) >> v13) & 1;
    else
      result = (*(_QWORD *)(*(_QWORD *)v14 + 8LL * ((unsigned int)v13 >> 6)) >> v13) & 1LL;
    if ( (_BYTE)result )
      goto LABEL_35;
    result = a7(a8, (unsigned int)v13);
    v15 = result;
    if ( !(_DWORD)result )
      goto LABEL_35;
    v16 = *(int **)(a1 + 8);
    LODWORD(v17) = a3 - v12;
    v18 = (_DWORD *)(a2 + 4LL * v12);
    if ( a3 - v12 > (unsigned int)a5 )
      LODWORD(v17) = a5;
    v19 = 4LL * (unsigned int)v17;
    v20 = &v18[(unsigned __int64)v19 / 4];
    v21 = v19 >> 2;
    v22 = v19 >> 4;
    if ( v22 )
    {
      v23 = *v16;
      result = a2 + 4LL * v12;
      v24 = &v18[4 * v22];
      while ( *(_DWORD *)result == v23 )
      {
        if ( v23 != *(_DWORD *)(result + 4) )
        {
          result += 4;
          goto LABEL_16;
        }
        if ( v23 != *(_DWORD *)(result + 8) )
        {
          result += 8;
          goto LABEL_16;
        }
        if ( v23 != *(_DWORD *)(result + 12) )
        {
          result += 12;
          goto LABEL_16;
        }
        result += 16;
        if ( v24 == (_DWORD *)result )
        {
          v21 = ((__int64)v20 - result) >> 2;
          goto LABEL_52;
        }
      }
      goto LABEL_16;
    }
    result = a2 + 4LL * v12;
LABEL_52:
    if ( v21 == 2 )
    {
      v23 = *v16;
      goto LABEL_59;
    }
    if ( v21 != 3 )
    {
      if ( v21 != 1 )
        goto LABEL_17;
      v23 = *v16;
LABEL_56:
      if ( v23 == *(_DWORD *)result )
        goto LABEL_17;
      goto LABEL_16;
    }
    v23 = *v16;
    if ( *(_DWORD *)result == *v16 )
    {
      result += 4;
LABEL_59:
      if ( *(_DWORD *)result == v23 )
      {
        result += 4;
        goto LABEL_56;
      }
    }
LABEL_16:
    if ( v20 != (_DWORD *)result )
    {
      while ( v18 != v20 )
        *v18++ = v23;
      v44 = *(__int64 **)a1;
      v45 = **(_QWORD **)a1;
      if ( (v45 & 1) == 0 )
        goto LABEL_34;
      goto LABEL_40;
    }
LABEL_17:
    v17 = (int)v17;
    if ( !(_DWORD)v17 )
      goto LABEL_35;
    v53 = *(int **)(a1 + 8);
    v25 = a1;
    v26 = 0x7FFFFFFF;
    v27 = v20;
    v28 = 0;
    v29 = (_DWORD *)(a2 + 4LL * v12);
    v30 = v25;
    do
    {
      v31 = v12 + (int)v28;
      v32 = *(_DWORD *)(a4 + 4 * v31);
      if ( v32 == -1 )
      {
        v47 = *(unsigned __int8 **)(**(_QWORD **)(v30 + 16) + 8 * v31);
        if ( (unsigned __int8)sub_2B0D8B0(v47) && *v47 != 13 )
        {
LABEL_44:
          v48 = v30;
          v49 = v29;
          v50 = v27;
          a1 = v48;
          for ( i = *v53; v49 != v50; ++v49 )
            *v49 = i;
          v44 = *(__int64 **)a1;
          v45 = **(_QWORD **)a1;
          if ( (v45 & 1) != 0 )
            goto LABEL_40;
          v52 = *(_QWORD *)v45;
          result = 1LL << v13;
          *(_QWORD *)(v52 + 8LL * ((unsigned int)v13 >> 6)) |= 1LL << v13;
          goto LABEL_35;
        }
      }
      else
      {
        if ( v15 <= v32 )
          goto LABEL_44;
        if ( v26 > v32 )
          v26 = *(_DWORD *)(a4 + 4 * v31);
      }
      ++v28;
    }
    while ( v17 != v28 );
    v33 = v30;
    v34 = v29;
    v35 = v27;
    v36 = 0;
    a1 = v33;
    v54 = v34;
    result = (unsigned int)(a5 * (v26 / a5));
    while ( 1 )
    {
      v39 = v12 + v36;
      v40 = *(_DWORD *)(a4 + 4LL * (v12 + (int)v36));
      if ( v40 != -1 )
        break;
LABEL_28:
      if ( v17 == ++v36 )
        goto LABEL_35;
    }
    v41 = v40 - result;
    if ( a5 > v41 )
    {
      v37 = v12 + v41;
      v38 = (unsigned int *)(a2 + 4LL * v37);
      if ( *v38 > v39 && *v38 != v37 )
        *v38 = v39;
      goto LABEL_28;
    }
    v42 = v54;
    for ( j = **(_DWORD **)(a1 + 8); v42 != v35; ++v42 )
      *v42 = j;
    v44 = *(__int64 **)a1;
    v45 = **(_QWORD **)a1;
    if ( (v45 & 1) == 0 )
    {
LABEL_34:
      v46 = *(_QWORD *)v45;
      result = 1LL << v13;
      *(_QWORD *)(v46 + 8LL * ((unsigned int)v13 >> 6)) |= 1LL << v13;
      goto LABEL_35;
    }
LABEL_40:
    result = 2 * ((v45 >> 58 << 57) | ~(-1LL << (v45 >> 58)) & (~(-1LL << (v45 >> 58)) & (v45 >> 1) | (1LL << v13))) + 1;
    *v44 = result;
LABEL_35:
    ++v13;
    v12 += a5;
  }
  while ( v57 != v13 );
  return result;
}

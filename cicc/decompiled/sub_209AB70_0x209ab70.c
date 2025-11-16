// Function: sub_209AB70
// Address: 0x209ab70
//
__int64 __fastcall sub_209AB70(__int64 *a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v6; // rdx
  __int64 *v7; // r12
  __int64 result; // rax
  __int64 v9; // rax
  __int64 v10; // r14
  int v11; // r8d
  __int64 v12; // rsi
  int v13; // r9d
  unsigned int i; // ecx
  __int64 v15; // rdx
  unsigned int v16; // ecx
  __int64 v17; // rax
  char *v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v20; // r8
  unsigned __int64 v21; // rdi
  __int64 v22; // rcx
  char *v23; // rax
  char *v24; // rcx
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // rdx
  _QWORD *v28; // r13
  __int64 v29; // rax
  unsigned int v30; // edx
  unsigned __int8 v31; // al
  _QWORD *v32; // rax
  unsigned int v33; // esi
  _QWORD *v34; // r13
  int v35; // edx
  int v36; // r15d
  __int64 v37; // r8
  __int64 v38; // r10
  int v39; // r11d
  unsigned int j; // r9d
  unsigned int v41; // eax
  __int64 *v42; // rdx
  __int64 v43; // r8
  int v44; // eax
  int v45; // ecx
  unsigned __int64 v46; // rdx
  int v47; // edx
  int v48; // ecx
  __int64 v49; // [rsp+10h] [rbp-50h] BYREF
  int v50; // [rsp+18h] [rbp-48h] BYREF
  char v51; // [rsp+1Ch] [rbp-44h]
  __int64 *v52; // [rsp+20h] [rbp-40h] BYREF
  __int64 v53; // [rsp+28h] [rbp-38h]

  v7 = sub_20685E0(a2, a1, a3, a4, a5);
  result = (_WORD)v7[3] & 0xFFFB;
  if ( (v7[3] & 0xFFFB) == 0xA || (v7[3] & 0xFFFB) == 0x20 )
    return result;
  v9 = *(unsigned int *)(a2 + 272);
  v10 = v6;
  if ( (_DWORD)v9 )
  {
    v11 = v6;
    v12 = *(_QWORD *)(a2 + 256);
    v13 = 1;
    for ( i = (v9 - 1) & (v6 + (((unsigned __int64)v7 >> 9) ^ ((unsigned __int64)v7 >> 4))); ; i = (v9 - 1) & v16 )
    {
      v15 = v12 + 32LL * i;
      if ( v7 == *(__int64 **)v15 )
      {
        if ( *(_DWORD *)(v15 + 8) == v11 )
        {
          result = v12 + 32 * v9;
          if ( v15 != result && *(_QWORD *)(v15 + 16) )
            return result;
          break;
        }
      }
      else if ( !*(_QWORD *)v15 && *(_DWORD *)(v15 + 8) == -1 )
      {
        break;
      }
      v16 = v13 + i;
      ++v13;
    }
  }
  result = sub_209A520((__int64)&v50, (__int64)a1, a2, 6);
  if ( !v51 )
    return result;
  v17 = *(_QWORD *)(a2 + 712);
  v18 = *(char **)(v17 + 568);
  v19 = *(unsigned int *)(v17 + 576);
  v20 = 4 * v19;
  v21 = v19;
  if ( !(v19 >> 2) )
  {
    v24 = v18;
LABEL_42:
    v43 = &v18[v20] - v24;
    switch ( v43 )
    {
      case 8LL:
        v44 = v50;
        break;
      case 12LL:
        v44 = v50;
        if ( *(_DWORD *)v24 == v50 )
        {
LABEL_47:
          v21 = (v24 - v18) >> 2;
          goto LABEL_19;
        }
        v24 += 4;
        break;
      case 4LL:
        v44 = v50;
LABEL_46:
        if ( *(_DWORD *)v24 != v44 )
          goto LABEL_19;
        goto LABEL_47;
      default:
        goto LABEL_19;
    }
    if ( *(_DWORD *)v24 != v44 )
    {
      v24 += 4;
      goto LABEL_46;
    }
    goto LABEL_47;
  }
  v22 = 16 * (v19 >> 2);
  v23 = v18;
  v24 = &v18[v22];
  while ( 1 )
  {
    if ( *(_DWORD *)v23 == v50 )
      goto LABEL_18;
    if ( v50 == *((_DWORD *)v23 + 1) )
    {
      v23 += 4;
LABEL_18:
      v21 = (v23 - v18) >> 2;
      goto LABEL_19;
    }
    if ( v50 == *((_DWORD *)v23 + 2) )
    {
      v21 = (v23 + 8 - v18) >> 2;
      goto LABEL_19;
    }
    if ( v50 == *((_DWORD *)v23 + 3) )
      break;
    v23 += 16;
    if ( v23 == v24 )
      goto LABEL_42;
  }
  v21 = (v23 + 12 - v18) >> 2;
LABEL_19:
  v25 = *(_QWORD *)(a2 + 280);
  if ( (v25 & 1) != 0 )
  {
    v26 = v25 >> 58;
    v27 = ~(-1LL << (v25 >> 58));
    result = v27 & (v25 >> 1);
    if ( _bittest64(&result, v21) )
      return result;
    *(_QWORD *)(a2 + 280) = 2 * ((v26 << 57) | v27 & (result | (1LL << v21))) + 1;
  }
  else
  {
    v42 = (__int64 *)(*(_QWORD *)v25 + 8LL * ((unsigned int)v21 >> 6));
    result = *v42;
    if ( _bittest64(&result, v21) )
      return result;
    *v42 = result | (1LL << (v21 & 0x3F));
  }
  v28 = *(_QWORD **)(a2 + 552);
  v29 = sub_1E0A0C0(v28[4]);
  v30 = 8 * sub_15A9520(v29, *(_DWORD *)(v29 + 4));
  if ( v30 == 32 )
  {
    v31 = 5;
  }
  else if ( v30 > 0x20 )
  {
    v31 = 6;
    if ( v30 != 64 )
    {
      v31 = 0;
      if ( v30 == 128 )
        v31 = 7;
    }
  }
  else
  {
    v31 = 3;
    if ( v30 != 8 )
      v31 = 4 * (v30 == 16);
  }
  v32 = sub_1D299D0(v28, v50, v31, 0, 1);
  v33 = *(_DWORD *)(a2 + 272);
  v53 = v10;
  v52 = v7;
  v34 = v32;
  v36 = v35;
  if ( !v33 )
  {
    ++*(_QWORD *)(a2 + 248);
    goto LABEL_60;
  }
  v37 = *(_QWORD *)(a2 + 256);
  v38 = 0;
  v39 = 1;
  for ( j = (v33 - 1) & (v53 + (((unsigned __int64)v7 >> 9) ^ ((unsigned __int64)v7 >> 4))); ; j = (v33 - 1) & v41 )
  {
    result = v37 + 32LL * j;
    if ( v7 != *(__int64 **)result )
      break;
    if ( (_DWORD)v53 == *(_DWORD *)(result + 8) )
      goto LABEL_52;
LABEL_30:
    v41 = j + v39++;
  }
  if ( *(_QWORD *)result )
    goto LABEL_30;
  v47 = *(_DWORD *)(result + 8);
  if ( v47 != -1 )
  {
    if ( v47 == -2 && !v38 )
      v38 = v37 + 32LL * j;
    goto LABEL_30;
  }
  v48 = *(_DWORD *)(a2 + 264);
  if ( v38 )
    result = v38;
  ++*(_QWORD *)(a2 + 248);
  v45 = v48 + 1;
  if ( 4 * v45 >= 3 * v33 )
  {
LABEL_60:
    v33 *= 2;
    goto LABEL_61;
  }
  if ( v33 - *(_DWORD *)(a2 + 268) - v45 <= v33 >> 3 )
  {
LABEL_61:
    sub_2099180(a2 + 248, v33);
    sub_2098DD0(a2 + 248, (unsigned __int64 *)&v52, &v49);
    result = v49;
    v45 = *(_DWORD *)(a2 + 264) + 1;
  }
  *(_DWORD *)(a2 + 264) = v45;
  if ( *(_QWORD *)result || *(_DWORD *)(result + 8) != -1 )
    --*(_DWORD *)(a2 + 268);
  v46 = (unsigned __int64)v52;
  *(_DWORD *)(result + 24) = 0;
  *(_QWORD *)result = v46;
  LODWORD(v46) = v53;
  *(_QWORD *)(result + 16) = 0;
  *(_DWORD *)(result + 8) = v46;
LABEL_52:
  *(_QWORD *)(result + 16) = v34;
  *(_DWORD *)(result + 24) = v36;
  return result;
}

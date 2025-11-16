// Function: sub_2644120
// Address: 0x2644120
//
_QWORD *__fastcall sub_2644120(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rdx
  int v9; // r8d
  __int64 v10; // rsi
  _QWORD *v11; // rdi
  int v12; // ecx
  __int64 v13; // r11
  unsigned int v14; // edx
  __int64 v15; // rbx
  int v16; // r12d
  int v17; // r12d
  __int64 v18; // r11
  unsigned int v19; // edx
  __int64 v20; // rbx
  int v21; // r12d
  __int64 v22; // r11
  unsigned int v23; // edx
  __int64 v24; // rbx
  int v25; // r12d
  __int64 v26; // r11
  unsigned int v27; // edx
  __int64 v28; // rbx
  __int64 v29; // rsi
  __int64 v30; // rdi
  int v31; // edx
  unsigned int v32; // ecx
  __int64 v33; // r8
  int v34; // r11d
  __int64 v35; // rdi
  int v36; // edx
  unsigned int v37; // ecx
  __int64 v38; // r8
  int v39; // r10d
  __int64 v40; // rdi
  unsigned int v41; // ecx
  __int64 v42; // r8
  int v43; // r11d

  result = a1;
  v6 = a2 - (_QWORD)a1;
  v7 = (a2 - (__int64)a1) >> 6;
  v8 = v6 >> 4;
  if ( v7 > 0 )
  {
    v9 = *(_DWORD *)(a3 + 24);
    v10 = *(_QWORD *)(a3 + 8);
    v11 = &result[8 * v7];
    v12 = v9 - 1;
    while ( 1 )
    {
      v13 = *(_QWORD *)(*result + 8LL);
      if ( v9 )
      {
        v14 = v12 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v15 = *(_QWORD *)(v10 + 24LL * v14);
        if ( v13 == v15 )
          return result;
        v17 = 1;
        while ( v15 != -4096 )
        {
          v14 = v12 & (v17 + v14);
          v15 = *(_QWORD *)(v10 + 24LL * v14);
          if ( v13 == v15 )
            return result;
          ++v17;
        }
        v18 = *(_QWORD *)(result[2] + 8LL);
        v19 = v12 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v20 = *(_QWORD *)(v10 + 24LL * v19);
        if ( v20 == v18 )
        {
LABEL_17:
          result += 2;
          return result;
        }
        v21 = 1;
        while ( v20 != -4096 )
        {
          v19 = v12 & (v21 + v19);
          v20 = *(_QWORD *)(v10 + 24LL * v19);
          if ( v20 == v18 )
            goto LABEL_17;
          ++v21;
        }
        v22 = *(_QWORD *)(result[4] + 8LL);
        v23 = v12 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v24 = *(_QWORD *)(v10 + 24LL * v23);
        if ( v22 == v24 )
        {
LABEL_21:
          result += 4;
          return result;
        }
        v25 = 1;
        while ( v24 != -4096 )
        {
          v23 = v12 & (v25 + v23);
          v24 = *(_QWORD *)(v10 + 24LL * v23);
          if ( v22 == v24 )
            goto LABEL_21;
          ++v25;
        }
        v26 = *(_QWORD *)(result[6] + 8LL);
        v27 = v12 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v28 = *(_QWORD *)(v10 + 24LL * v27);
        if ( v28 == v26 )
        {
LABEL_25:
          result += 6;
          return result;
        }
        v16 = 1;
        while ( v28 != -4096 )
        {
          v27 = v12 & (v16 + v27);
          v28 = *(_QWORD *)(v10 + 24LL * v27);
          if ( v28 == v26 )
            goto LABEL_25;
          ++v16;
        }
      }
      result += 8;
      if ( v11 == result )
      {
        v8 = (a2 - (__int64)result) >> 4;
        break;
      }
    }
  }
  switch ( v8 )
  {
    case 2LL:
      v29 = *(_QWORD *)(a3 + 8);
      v31 = *(_DWORD *)(a3 + 24);
      break;
    case 3LL:
      v29 = *(_QWORD *)(a3 + 8);
      v30 = *(_QWORD *)(*result + 8LL);
      v31 = *(_DWORD *)(a3 + 24);
      if ( v31 )
      {
        v32 = (v31 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v33 = *(_QWORD *)(v29 + 24LL * v32);
        if ( v30 == v33 )
          return result;
        v34 = 1;
        while ( v33 != -4096 )
        {
          v32 = (v31 - 1) & (v34 + v32);
          v33 = *(_QWORD *)(v29 + 24LL * v32);
          if ( v30 == v33 )
            return result;
          ++v34;
        }
      }
      result += 2;
      break;
    case 1LL:
      v29 = *(_QWORD *)(a3 + 8);
      v31 = *(_DWORD *)(a3 + 24);
      goto LABEL_34;
    default:
      return (_QWORD *)a2;
  }
  v40 = *(_QWORD *)(*result + 8LL);
  if ( v31 )
  {
    v41 = (v31 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
    v42 = *(_QWORD *)(v29 + 24LL * v41);
    if ( v42 == v40 )
      return result;
    v43 = 1;
    while ( v42 != -4096 )
    {
      v41 = (v31 - 1) & (v43 + v41);
      v42 = *(_QWORD *)(v29 + 24LL * v41);
      if ( v40 == v42 )
        return result;
      ++v43;
    }
  }
  result += 2;
LABEL_34:
  v35 = *(_QWORD *)(*result + 8LL);
  if ( !v31 )
    return (_QWORD *)a2;
  v36 = v31 - 1;
  v37 = v36 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
  v38 = *(_QWORD *)(v29 + 24LL * v37);
  if ( v35 != v38 )
  {
    v39 = 1;
    while ( v38 != -4096 )
    {
      v37 = v36 & (v39 + v37);
      v38 = *(_QWORD *)(v29 + 24LL * v37);
      if ( v35 == v38 )
        return result;
      ++v39;
    }
    return (_QWORD *)a2;
  }
  return result;
}

// Function: sub_19965D0
// Address: 0x19965d0
//
_QWORD *__fastcall sub_19965D0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  __int64 v6; // r8
  __int64 v7; // rdx
  int v8; // r9d
  __int64 v9; // rsi
  _QWORD *v10; // r8
  int v11; // ecx
  unsigned int v12; // edx
  __int64 v13; // rbx
  int v14; // r13d
  __int64 v15; // rdi
  _QWORD *v16; // r12
  unsigned int v17; // edx
  __int64 v18; // rbx
  int v19; // r13d
  __int64 v20; // rdi
  unsigned int v21; // edx
  __int64 v22; // rbx
  int v23; // r13d
  int v24; // r12d
  __int64 v25; // rdi
  unsigned int v26; // edx
  __int64 v27; // rbx
  int v28; // edx
  __int64 v29; // rsi
  unsigned int v30; // ecx
  __int64 v31; // r8
  int v32; // r11d
  int v33; // edx
  int v34; // r9d
  unsigned int v35; // ecx
  __int64 v36; // r8
  unsigned int v37; // ecx
  __int64 v38; // r8
  int v39; // r11d

  result = a1;
  v6 = (a2 - (__int64)a1) >> 5;
  v7 = (a2 - (__int64)a1) >> 3;
  if ( v6 > 0 )
  {
    v8 = *(_DWORD *)(a3 + 24);
    v9 = *(_QWORD *)(a3 + 8);
    v10 = &a1[4 * v6];
    v11 = v8 - 1;
    while ( 1 )
    {
      if ( v8 )
      {
        v12 = v11 & (((unsigned int)*result >> 9) ^ ((unsigned int)*result >> 4));
        v13 = *(_QWORD *)(v9 + 8LL * v12);
        if ( *result == v13 )
          return result;
        v24 = 1;
        while ( v13 != -8 )
        {
          v12 = v11 & (v24 + v12);
          v13 = *(_QWORD *)(v9 + 8LL * v12);
          if ( *result == v13 )
            return result;
          ++v24;
        }
        v25 = result[1];
        v16 = result + 1;
        v26 = v11 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v27 = *(_QWORD *)(v9 + 8LL * v26);
        if ( v27 == v25 )
          return v16;
        v14 = 1;
        while ( v27 != -8 )
        {
          v26 = v11 & (v14 + v26);
          v27 = *(_QWORD *)(v9 + 8LL * v26);
          if ( v27 == v25 )
            return v16;
          ++v14;
        }
        v15 = result[2];
        v16 = result + 2;
        v17 = v11 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v18 = *(_QWORD *)(v9 + 8LL * v17);
        if ( v15 == v18 )
          return v16;
        v19 = 1;
        while ( v18 != -8 )
        {
          v17 = v11 & (v19 + v17);
          v18 = *(_QWORD *)(v9 + 8LL * v17);
          if ( v18 == v15 )
            return v16;
          ++v19;
        }
        v20 = result[3];
        v16 = result + 3;
        v21 = v11 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v22 = *(_QWORD *)(v9 + 8LL * v21);
        if ( v20 == v22 )
          return v16;
        v23 = 1;
        while ( v22 != -8 )
        {
          v21 = v11 & (v23 + v21);
          v22 = *(_QWORD *)(v9 + 8LL * v21);
          if ( v22 == v20 )
            return v16;
          ++v23;
        }
      }
      result += 4;
      if ( result == v10 )
      {
        v7 = (a2 - (__int64)result) >> 3;
        break;
      }
    }
  }
  switch ( v7 )
  {
    case 2LL:
      v29 = *(_QWORD *)(a3 + 8);
      v28 = *(_DWORD *)(a3 + 24);
      break;
    case 3LL:
      v28 = *(_DWORD *)(a3 + 24);
      v29 = *(_QWORD *)(a3 + 8);
      if ( v28 )
      {
        v30 = (v28 - 1) & (((unsigned int)*result >> 9) ^ ((unsigned int)*result >> 4));
        v31 = *(_QWORD *)(v29 + 8LL * v30);
        if ( *result == v31 )
          return result;
        v32 = 1;
        while ( v31 != -8 )
        {
          v30 = (v28 - 1) & (v32 + v30);
          v31 = *(_QWORD *)(v29 + 8LL * v30);
          if ( *result == v31 )
            return result;
          ++v32;
        }
      }
      ++result;
      break;
    case 1LL:
      v29 = *(_QWORD *)(a3 + 8);
      v28 = *(_DWORD *)(a3 + 24);
      goto LABEL_30;
    default:
      return (_QWORD *)a2;
  }
  if ( v28 )
  {
    v37 = (v28 - 1) & (((unsigned int)*result >> 9) ^ ((unsigned int)*result >> 4));
    v38 = *(_QWORD *)(v29 + 8LL * v37);
    if ( *result == v38 )
      return result;
    v39 = 1;
    while ( v38 != -8 )
    {
      v37 = (v28 - 1) & (v39 + v37);
      v38 = *(_QWORD *)(v29 + 8LL * v37);
      if ( *result == v38 )
        return result;
      ++v39;
    }
  }
  ++result;
LABEL_30:
  if ( !v28 )
    return (_QWORD *)a2;
  v33 = v28 - 1;
  v34 = 1;
  v35 = v33 & (((unsigned int)*result >> 9) ^ ((unsigned int)*result >> 4));
  v36 = *(_QWORD *)(v29 + 8LL * v35);
  if ( *result != v36 )
  {
    while ( v36 != -8 )
    {
      v35 = v33 & (v34 + v35);
      v36 = *(_QWORD *)(v29 + 8LL * v35);
      if ( *result == v36 )
        return result;
      ++v34;
    }
    return (_QWORD *)a2;
  }
  return result;
}

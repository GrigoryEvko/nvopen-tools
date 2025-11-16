// Function: sub_142C880
// Address: 0x142c880
//
_QWORD *__fastcall sub_142C880(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  __int64 v6; // r8
  __int64 v7; // rdx
  int v8; // r9d
  __int64 v9; // rsi
  _QWORD *v10; // r8
  int v11; // ecx
  __int64 v12; // rdx
  unsigned int v13; // edi
  __int64 v14; // rbx
  int v15; // r13d
  _QWORD *v16; // r12
  __int64 v17; // rdx
  unsigned int v18; // ebx
  __int64 v19; // rdi
  int v20; // r13d
  __int64 v21; // rdx
  unsigned int v22; // ebx
  __int64 v23; // rdi
  int v24; // r13d
  int v25; // r12d
  __int64 v26; // rdx
  unsigned int v27; // edi
  __int64 v28; // rbx
  int v29; // edx
  __int64 v30; // rsi
  __int64 v31; // rdi
  unsigned int v32; // ecx
  __int64 v33; // r8
  int v34; // r11d
  int v35; // edx
  int v36; // r9d
  __int64 v37; // rcx
  unsigned int v38; // edi
  __int64 v39; // r8
  __int64 v40; // rcx
  unsigned int v41; // edi
  __int64 v42; // r8
  int v43; // r11d

  result = a1;
  v6 = (a2 - (__int64)a1) >> 6;
  v7 = (a2 - (__int64)a1) >> 4;
  if ( v6 > 0 )
  {
    v8 = *(_DWORD *)(a3 + 24);
    v9 = *(_QWORD *)(a3 + 8);
    v10 = &a1[8 * v6];
    v11 = v8 - 1;
    while ( 1 )
    {
      if ( v8 )
      {
        v12 = *(_QWORD *)(*result & 0xFFFFFFFFFFFFFFF8LL);
        v13 = v11 & (37 * v12);
        v14 = *(_QWORD *)(v9 + 8LL * v13);
        if ( v12 == v14 )
          return result;
        v25 = 1;
        while ( v14 != -1 )
        {
          v13 = v11 & (v25 + v13);
          v14 = *(_QWORD *)(v9 + 8LL * v13);
          if ( v12 == v14 )
            return result;
          ++v25;
        }
        v16 = result + 2;
        v26 = *(_QWORD *)(result[2] & 0xFFFFFFFFFFFFFFF8LL);
        v27 = v11 & (37 * v26);
        v28 = *(_QWORD *)(v9 + 8LL * v27);
        if ( v26 == v28 )
          return v16;
        v15 = 1;
        while ( v28 != -1 )
        {
          v27 = v11 & (v15 + v27);
          v28 = *(_QWORD *)(v9 + 8LL * v27);
          if ( v28 == v26 )
            return v16;
          ++v15;
        }
        v16 = result + 4;
        v17 = *(_QWORD *)(result[4] & 0xFFFFFFFFFFFFFFF8LL);
        v18 = v11 & (37 * v17);
        v19 = *(_QWORD *)(v9 + 8LL * v18);
        if ( v17 == v19 )
          return v16;
        v20 = 1;
        while ( v19 != -1 )
        {
          v18 = v11 & (v20 + v18);
          v19 = *(_QWORD *)(v9 + 8LL * v18);
          if ( v17 == v19 )
            return v16;
          ++v20;
        }
        v16 = result + 6;
        v21 = *(_QWORD *)(result[6] & 0xFFFFFFFFFFFFFFF8LL);
        v22 = v11 & (37 * v21);
        v23 = *(_QWORD *)(v9 + 8LL * v22);
        if ( v21 == v23 )
          return v16;
        v24 = 1;
        while ( v23 != -1 )
        {
          v22 = v11 & (v24 + v22);
          v23 = *(_QWORD *)(v9 + 8LL * v22);
          if ( v23 == v21 )
            return v16;
          ++v24;
        }
      }
      result += 8;
      if ( v10 == result )
      {
        v7 = (a2 - (__int64)result) >> 4;
        break;
      }
    }
  }
  switch ( v7 )
  {
    case 2LL:
      v30 = *(_QWORD *)(a3 + 8);
      v29 = *(_DWORD *)(a3 + 24);
      break;
    case 3LL:
      v29 = *(_DWORD *)(a3 + 24);
      v30 = *(_QWORD *)(a3 + 8);
      if ( v29 )
      {
        v31 = *(_QWORD *)(*result & 0xFFFFFFFFFFFFFFF8LL);
        v32 = (v29 - 1) & (37 * v31);
        v33 = *(_QWORD *)(v30 + 8LL * v32);
        if ( v31 == v33 )
          return result;
        v34 = 1;
        while ( v33 != -1 )
        {
          v32 = (v29 - 1) & (v34 + v32);
          v33 = *(_QWORD *)(v30 + 8LL * v32);
          if ( v31 == v33 )
            return result;
          ++v34;
        }
      }
      result += 2;
      break;
    case 1LL:
      v30 = *(_QWORD *)(a3 + 8);
      v29 = *(_DWORD *)(a3 + 24);
      goto LABEL_30;
    default:
      return (_QWORD *)a2;
  }
  if ( v29 )
  {
    v40 = *(_QWORD *)(*result & 0xFFFFFFFFFFFFFFF8LL);
    v41 = (v29 - 1) & (37 * v40);
    v42 = *(_QWORD *)(v30 + 8LL * v41);
    if ( v40 == v42 )
      return result;
    v43 = 1;
    while ( v42 != -1 )
    {
      v41 = (v29 - 1) & (v43 + v41);
      v42 = *(_QWORD *)(v30 + 8LL * v41);
      if ( v40 == v42 )
        return result;
      ++v43;
    }
  }
  result += 2;
LABEL_30:
  if ( !v29 )
    return (_QWORD *)a2;
  v35 = v29 - 1;
  v36 = 1;
  v37 = *(_QWORD *)(*result & 0xFFFFFFFFFFFFFFF8LL);
  v38 = v35 & (37 * v37);
  v39 = *(_QWORD *)(v30 + 8LL * v38);
  if ( v37 != v39 )
  {
    while ( v39 != -1 )
    {
      v38 = v35 & (v36 + v38);
      v39 = *(_QWORD *)(v30 + 8LL * v38);
      if ( v37 == v39 )
        return result;
      ++v36;
    }
    return (_QWORD *)a2;
  }
  return result;
}

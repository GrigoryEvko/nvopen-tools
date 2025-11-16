// Function: sub_1A26660
// Address: 0x1a26660
//
__int64 *__fastcall sub_1A26660(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r12
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 *v8; // r14
  __int64 v9; // rax
  int v10; // edx
  int v11; // ecx
  __int64 v12; // r9
  unsigned int v13; // edx
  __int64 *v14; // rdi
  __int64 v15; // r8
  __int64 v17; // rax
  int v18; // edx
  __int64 v19; // rsi
  int v20; // ecx
  __int64 v21; // r9
  unsigned int v22; // edx
  __int64 *v23; // rdi
  __int64 v24; // r8
  int v25; // edi
  int v26; // edx
  __int64 v27; // rsi
  int v28; // ecx
  __int64 v29; // r9
  unsigned int v30; // edx
  __int64 v31; // r8
  int v32; // edi
  int v33; // r10d
  int v34; // edx
  __int64 v35; // rsi
  int v36; // ecx
  __int64 v37; // r9
  unsigned int v38; // edx
  __int64 v39; // r8
  int v40; // edi
  int v41; // r10d
  int v42; // edi
  int v43; // r10d
  int v44; // r10d
  __int64 v45; // [rsp+0h] [rbp-30h] BYREF
  __int64 v46; // [rsp+8h] [rbp-28h]

  v4 = a1;
  v5 = (a2 - (__int64)a1) >> 5;
  v6 = (a2 - (__int64)a1) >> 3;
  v45 = a3;
  v46 = a4;
  if ( v5 <= 0 )
  {
LABEL_31:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          return (__int64 *)a2;
LABEL_39:
        if ( !(unsigned __int8)sub_1A26430(&v45, v4) )
          return (__int64 *)a2;
        return v4;
      }
      if ( (unsigned __int8)sub_1A26430(&v45, v4) )
        return v4;
      ++v4;
    }
    if ( (unsigned __int8)sub_1A26430(&v45, v4) )
      return v4;
    ++v4;
    goto LABEL_39;
  }
  v7 = &a1[4 * v5];
  while ( !sub_1A26350(v45, *v4) )
  {
    v8 = v4 + 1;
    if ( sub_1A26350(v45, v4[1]) )
    {
      v17 = v46;
      v18 = *(_DWORD *)(v46 + 24);
      if ( !v18 )
        return v8;
      v19 = v4[1];
      v20 = v18 - 1;
      v21 = *(_QWORD *)(v46 + 8);
      v22 = (v18 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v23 = (__int64 *)(v21 + 8LL * v22);
      v24 = *v23;
      if ( v19 == *v23 )
        goto LABEL_14;
      v25 = 1;
      while ( v24 != -8 )
      {
        v44 = v25 + 1;
        v22 = v20 & (v25 + v22);
        v23 = (__int64 *)(v21 + 8LL * v22);
        v24 = *v23;
        if ( v19 == *v23 )
          goto LABEL_14;
        v25 = v44;
      }
      return v8;
    }
    v8 = v4 + 2;
    if ( sub_1A26350(v45, v4[2]) )
    {
      v17 = v46;
      v26 = *(_DWORD *)(v46 + 24);
      if ( !v26 )
        return v8;
      v27 = v4[2];
      v28 = v26 - 1;
      v29 = *(_QWORD *)(v46 + 8);
      v30 = (v26 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v23 = (__int64 *)(v29 + 8LL * v30);
      v31 = *v23;
      if ( v27 == *v23 )
        goto LABEL_14;
      v32 = 1;
      while ( v31 != -8 )
      {
        v33 = v32 + 1;
        v30 = v28 & (v32 + v30);
        v23 = (__int64 *)(v29 + 8LL * v30);
        v31 = *v23;
        if ( v27 == *v23 )
          goto LABEL_14;
        v32 = v33;
      }
      return v8;
    }
    v8 = v4 + 3;
    if ( sub_1A26350(v45, v4[3]) )
    {
      v17 = v46;
      v34 = *(_DWORD *)(v46 + 24);
      if ( !v34 )
        return v8;
      v35 = v4[3];
      v36 = v34 - 1;
      v37 = *(_QWORD *)(v46 + 8);
      v38 = (v34 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v23 = (__int64 *)(v37 + 8LL * v38);
      v39 = *v23;
      if ( *v23 != v35 )
      {
        v40 = 1;
        while ( v39 != -8 )
        {
          v41 = v40 + 1;
          v38 = v36 & (v40 + v38);
          v23 = (__int64 *)(v37 + 8LL * v38);
          v39 = *v23;
          if ( v35 == *v23 )
            goto LABEL_14;
          v40 = v41;
        }
        return v8;
      }
LABEL_14:
      *v23 = -16;
      --*(_DWORD *)(v17 + 16);
      ++*(_DWORD *)(v17 + 20);
      return v8;
    }
    v4 += 4;
    if ( v7 == v4 )
    {
      v6 = (a2 - (__int64)v4) >> 3;
      goto LABEL_31;
    }
  }
  v9 = v46;
  v10 = *(_DWORD *)(v46 + 24);
  if ( v10 )
  {
    v11 = v10 - 1;
    v12 = *(_QWORD *)(v46 + 8);
    v13 = (v10 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
    v14 = (__int64 *)(v12 + 8LL * v13);
    v15 = *v14;
    if ( *v14 == *v4 )
    {
LABEL_10:
      *v14 = -16;
      --*(_DWORD *)(v9 + 16);
      ++*(_DWORD *)(v9 + 20);
    }
    else
    {
      v42 = 1;
      while ( v15 != -8 )
      {
        v43 = v42 + 1;
        v13 = v11 & (v42 + v13);
        v14 = (__int64 *)(v12 + 8LL * v13);
        v15 = *v14;
        if ( *v4 == *v14 )
          goto LABEL_10;
        v42 = v43;
      }
    }
  }
  return v4;
}

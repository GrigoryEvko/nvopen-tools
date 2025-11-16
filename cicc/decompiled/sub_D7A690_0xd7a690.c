// Function: sub_D7A690
// Address: 0xd7a690
//
__int64 __fastcall sub_D7A690(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  bool v6; // zf
  _QWORD *v7; // rax
  _QWORD *v8; // r13
  _QWORD *v9; // rbx
  _QWORD *v10; // r14
  __int64 result; // rax
  unsigned __int64 v12; // rbx
  _QWORD *v13; // rdx
  bool v14; // cl
  __int64 v15; // r13
  __int64 *v16; // rbx
  __int64 v17; // r14
  unsigned int v18; // esi
  int v19; // r11d
  _QWORD *v20; // rdx
  __int64 v21; // r8
  unsigned __int64 v22; // r15
  unsigned int v23; // edi
  __int64 v24; // rcx
  char *v25; // rsi
  __int64 v26; // rdi
  int v27; // eax
  int v28; // ecx
  int v29; // r9d
  int v30; // r9d
  __int64 v31; // r8
  __int64 v32; // r10
  int v33; // edi
  _QWORD *v34; // rsi
  int v35; // r8d
  int v36; // r8d
  __int64 v37; // r9
  int v38; // esi
  __int64 v39; // rdi
  unsigned __int64 v40; // [rsp+8h] [rbp-78h]
  __int64 v41; // [rsp+10h] [rbp-70h] BYREF
  __int64 v42; // [rsp+18h] [rbp-68h] BYREF
  __m128i v43; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v44; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v45; // [rsp+38h] [rbp-48h]
  __int64 v46; // [rsp+40h] [rbp-40h]

  v4 = *a3;
  *a3 = 0;
  v41 = v4;
  sub_B2F930(&v43, a2);
  v5 = sub_B2F650(v43.m128i_i64[0], v43.m128i_i64[1]);
  if ( (_QWORD **)v43.m128i_i64[0] != &v44 )
    j_j___libc_free_0(v43.m128i_i64[0], (char *)v44 + 1);
  v6 = *(_BYTE *)(a1 + 343) == 0;
  v42 = v5;
  if ( v6 )
  {
    v43.m128i_i64[1] = 0;
    v43.m128i_i64[0] = (__int64)byte_3F871B3;
  }
  else
  {
    v43.m128i_i64[0] = 0;
  }
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v7 = sub_9CA390((_QWORD *)a1, (unsigned __int64 *)&v42, &v43);
  v8 = v45;
  v9 = v44;
  v10 = v7;
  v40 = (unsigned __int64)(v7 + 4);
  if ( v45 != v44 )
  {
    do
    {
      if ( *v9 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v9 + 8LL))(*v9);
      ++v9;
    }
    while ( v8 != v9 );
    v9 = v44;
  }
  if ( v9 )
    j_j___libc_free_0(v9, v46 - (_QWORD)v9);
  result = v41;
  v10[5] = a2;
  v12 = *(unsigned __int8 *)(a1 + 343) | v40 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_DWORD *)(result + 8) == 1 )
  {
    v13 = *(_QWORD **)(result + 88);
    v14 = 0;
    if ( v13 )
      v14 = *v13 != v13[1];
    *(_BYTE *)(a1 + 347) |= v14;
  }
  v15 = *(_QWORD *)(result + 16);
  v16 = (__int64 *)(v12 & 0xFFFFFFFFFFFFFFF8LL);
  v17 = *v16;
  if ( v15 && v15 != v17 )
  {
    v18 = *(_DWORD *)(a1 + 328);
    if ( v18 )
    {
      v19 = 1;
      v20 = 0;
      v21 = *(_QWORD *)(a1 + 312);
      v22 = ((0xBF58476D1CE4E5B9LL * v15) >> 31) ^ (0xBF58476D1CE4E5B9LL * v15);
      v23 = v22 & (v18 - 1);
      result = v21 + 16LL * v23;
      v24 = *(_QWORD *)result;
      if ( v15 == *(_QWORD *)result )
      {
LABEL_20:
        if ( v17 != *(_QWORD *)(result + 8) )
          *(_QWORD *)(result + 8) = 0;
        goto LABEL_22;
      }
      while ( v24 != -1 )
      {
        if ( !v20 && v24 == -2 )
          v20 = (_QWORD *)result;
        v23 = (v18 - 1) & (v19 + v23);
        result = v21 + 16LL * v23;
        v24 = *(_QWORD *)result;
        if ( v15 == *(_QWORD *)result )
          goto LABEL_20;
        ++v19;
      }
      if ( !v20 )
        v20 = (_QWORD *)result;
      v27 = *(_DWORD *)(a1 + 320);
      ++*(_QWORD *)(a1 + 304);
      v28 = v27 + 1;
      if ( 4 * (v27 + 1) < 3 * v18 )
      {
        result = v18 - *(_DWORD *)(a1 + 324) - v28;
        if ( (unsigned int)result > v18 >> 3 )
        {
LABEL_41:
          *(_DWORD *)(a1 + 320) = v28;
          if ( *v20 != -1 )
            --*(_DWORD *)(a1 + 324);
          *v20 = v15;
          v20[1] = v17;
          goto LABEL_22;
        }
        sub_9D80B0(a1 + 304, v18);
        v35 = *(_DWORD *)(a1 + 328);
        if ( v35 )
        {
          v36 = v35 - 1;
          v37 = *(_QWORD *)(a1 + 312);
          v38 = 1;
          LODWORD(v22) = v36 & v22;
          v28 = *(_DWORD *)(a1 + 320) + 1;
          result = 0;
          v20 = (_QWORD *)(v37 + 16LL * (unsigned int)v22);
          v39 = *v20;
          if ( v15 != *v20 )
          {
            while ( v39 != -1 )
            {
              if ( !result && v39 == -2 )
                result = (__int64)v20;
              v22 = v36 & (unsigned int)(v22 + v38);
              v20 = (_QWORD *)(v37 + 16 * v22);
              v39 = *v20;
              if ( v15 == *v20 )
                goto LABEL_41;
              ++v38;
            }
            if ( result )
              v20 = (_QWORD *)result;
          }
          goto LABEL_41;
        }
LABEL_68:
        ++*(_DWORD *)(a1 + 320);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 304);
    }
    sub_9D80B0(a1 + 304, 2 * v18);
    v29 = *(_DWORD *)(a1 + 328);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 312);
      v28 = *(_DWORD *)(a1 + 320) + 1;
      result = v30 & ((unsigned int)((0xBF58476D1CE4E5B9LL * v15) >> 31) ^ (484763065 * (_DWORD)v15));
      v20 = (_QWORD *)(v31 + 16 * result);
      v32 = *v20;
      if ( v15 != *v20 )
      {
        v33 = 1;
        v34 = 0;
        while ( v32 != -1 )
        {
          if ( v32 == -2 && !v34 )
            v34 = v20;
          result = v30 & (unsigned int)(v33 + result);
          v20 = (_QWORD *)(v31 + 16LL * (unsigned int)result);
          v32 = *v20;
          if ( v15 == *v20 )
            goto LABEL_41;
          ++v33;
        }
        if ( v34 )
          v20 = v34;
      }
      goto LABEL_41;
    }
    goto LABEL_68;
  }
LABEL_22:
  v25 = (char *)v16[4];
  if ( v25 == (char *)v16[5] )
  {
    result = sub_9D0210(v16 + 3, v25, &v41);
    v26 = v41;
  }
  else
  {
    if ( v25 )
    {
      result = v41;
      *(_QWORD *)v25 = v41;
      v16[4] += 8;
      return result;
    }
    v16[4] = 8;
    v26 = v41;
  }
  if ( v26 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
  return result;
}

// Function: sub_3971E20
// Address: 0x3971e20
//
__int64 __fastcall sub_3971E20(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rsi
  __int64 v5; // rax
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 result; // rax
  __int64 v10; // rax
  _BYTE *v11; // rsi
  __int64 v12; // rdx
  _QWORD *v13; // r13
  const void *v14; // rdi
  size_t v15; // rdx
  __int64 v16; // rcx
  int v17; // eax
  unsigned int v18; // esi
  __int64 v19; // r8
  int v20; // r10d
  _QWORD *v21; // rdx
  unsigned int v22; // edi
  _QWORD *v23; // r13
  __int64 v24; // rcx
  int v25; // ecx
  int v26; // edi
  int v27; // edx
  int v28; // r9d
  int v29; // edx
  int v30; // r9d
  __int64 v31; // r8
  unsigned int v32; // ecx
  __int64 v33; // rsi
  int v34; // r11d
  _QWORD *v35; // r10
  int v36; // edx
  int v37; // esi
  __int64 v38; // r8
  _QWORD *v39; // r9
  unsigned int v40; // r15d
  int v41; // r10d
  __int64 v42; // rcx
  size_t v43; // [rsp-D0h] [rbp-D0h]
  __int64 v44; // [rsp-C8h] [rbp-C8h]
  __int64 v45; // [rsp-C0h] [rbp-C0h]
  __int64 v46; // [rsp-C0h] [rbp-C0h]
  __int64 v47; // [rsp-C0h] [rbp-C0h]
  __int64 v48; // [rsp-C0h] [rbp-C0h]
  __m128i v49; // [rsp-B8h] [rbp-B8h] BYREF
  __int16 v50; // [rsp-A8h] [rbp-A8h]
  __m128i v51; // [rsp-98h] [rbp-98h] BYREF
  char v52; // [rsp-88h] [rbp-88h]
  char v53; // [rsp-87h] [rbp-87h]
  __m128i v54[2]; // [rsp-78h] [rbp-78h] BYREF
  __int64 v55[2]; // [rsp-58h] [rbp-58h] BYREF
  _BYTE v56[72]; // [rsp-48h] [rbp-48h] BYREF

  if ( !*(_BYTE *)(a2 + 52) )
    return 0;
  v3 = *(_QWORD *)(a1 + 408);
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 8);
    v5 = *(unsigned int *)(v3 + 24);
  }
  else
  {
    v10 = sub_22077B0(0x20u);
    v3 = v10;
    if ( v10 )
    {
      *(_QWORD *)v10 = 0;
      *(_QWORD *)(v10 + 8) = 0;
      *(_QWORD *)(v10 + 16) = 0;
      *(_DWORD *)(v10 + 24) = 0;
      *(_QWORD *)(a1 + 408) = v10;
      goto LABEL_12;
    }
    v4 = MEMORY[8];
    v5 = MEMORY[0x18];
    *(_QWORD *)(a1 + 408) = 0;
  }
  if ( (_DWORD)v5 )
  {
    v6 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v4 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_6:
      if ( v7 != (__int64 *)(v4 + 16 * v5) )
        return v7[1];
    }
    else
    {
      v27 = 1;
      while ( v8 != -8 )
      {
        v28 = v27 + 1;
        v6 = (v5 - 1) & (v27 + v6);
        v7 = (__int64 *)(v4 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_6;
        v27 = v28;
      }
    }
  }
LABEL_12:
  v11 = *(_BYTE **)(a2 + 8);
  v12 = *(_QWORD *)(a2 + 16);
  v55[0] = (__int64)v56;
  sub_396BEA0(v55, v11, (__int64)&v11[v12]);
  v13 = (_QWORD *)sub_39B0D60();
  if ( !v13 )
  {
LABEL_16:
    v50 = 260;
    v53 = 1;
    v49.m128i_i64[0] = (__int64)v55;
    v51.m128i_i64[0] = (__int64)"no GCMetadataPrinter registered for GC: ";
    v52 = 3;
    sub_14EC200(v54, &v51, &v49);
    sub_16BCFB0((__int64)v54, 1u);
  }
  v14 = (const void *)v55[0];
  v15 = v55[1];
  while ( 1 )
  {
    v16 = v13[1];
    if ( v15 == *(_QWORD *)(v16 + 8) )
    {
      if ( !v15 )
        break;
      v11 = *(_BYTE **)v16;
      v43 = v15;
      v44 = v13[1];
      v17 = memcmp(v14, *(const void **)v16, v15);
      v16 = v44;
      v15 = v43;
      if ( !v17 )
        break;
    }
    v13 = (_QWORD *)*v13;
    if ( !v13 )
      goto LABEL_16;
  }
  (*(void (__fastcall **)(__m128i *, _BYTE *, size_t))(v16 + 32))(v54, v11, v15);
  result = v54[0].m128i_i64[0];
  *(_QWORD *)(v54[0].m128i_i64[0] + 8) = a2;
  v54[0].m128i_i64[0] = 0;
  v18 = *(_DWORD *)(v3 + 24);
  if ( !v18 )
  {
    ++*(_QWORD *)v3;
    goto LABEL_44;
  }
  v19 = *(_QWORD *)(v3 + 8);
  v20 = 1;
  v21 = 0;
  v22 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v23 = (_QWORD *)(v19 + 16LL * v22);
  v24 = *v23;
  if ( a2 != *v23 )
  {
    while ( v24 != -8 )
    {
      if ( v24 == -16 && !v21 )
        v21 = v23;
      v22 = (v18 - 1) & (v20 + v22);
      v23 = (_QWORD *)(v19 + 16LL * v22);
      v24 = *v23;
      if ( a2 == *v23 )
        goto LABEL_21;
      ++v20;
    }
    v25 = *(_DWORD *)(v3 + 16);
    if ( !v21 )
      v21 = v23;
    ++*(_QWORD *)v3;
    v26 = v25 + 1;
    if ( 4 * (v25 + 1) < 3 * v18 )
    {
      if ( v18 - *(_DWORD *)(v3 + 20) - v26 > v18 >> 3 )
      {
LABEL_36:
        *(_DWORD *)(v3 + 16) = v26;
        if ( *v21 != -8 )
          --*(_DWORD *)(v3 + 20);
        *v21 = a2;
        v21[1] = result;
        goto LABEL_22;
      }
      v48 = result;
      sub_3971C40(v3, v18);
      v36 = *(_DWORD *)(v3 + 24);
      if ( v36 )
      {
        v37 = v36 - 1;
        v38 = *(_QWORD *)(v3 + 8);
        v39 = 0;
        v40 = (v36 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v41 = 1;
        v26 = *(_DWORD *)(v3 + 16) + 1;
        result = v48;
        v21 = (_QWORD *)(v38 + 16LL * v40);
        v42 = *v21;
        if ( a2 != *v21 )
        {
          while ( v42 != -8 )
          {
            if ( !v39 && v42 == -16 )
              v39 = v21;
            v40 = v37 & (v41 + v40);
            v21 = (_QWORD *)(v38 + 16LL * v40);
            v42 = *v21;
            if ( a2 == *v21 )
              goto LABEL_36;
            ++v41;
          }
          if ( v39 )
            v21 = v39;
        }
        goto LABEL_36;
      }
LABEL_68:
      ++*(_DWORD *)(v3 + 16);
      BUG();
    }
LABEL_44:
    v47 = result;
    sub_3971C40(v3, 2 * v18);
    v29 = *(_DWORD *)(v3 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(v3 + 8);
      v32 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v26 = *(_DWORD *)(v3 + 16) + 1;
      result = v47;
      v21 = (_QWORD *)(v31 + 16LL * v32);
      v33 = *v21;
      if ( a2 != *v21 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -8 )
        {
          if ( !v35 && v33 == -16 )
            v35 = v21;
          v32 = v30 & (v34 + v32);
          v21 = (_QWORD *)(v31 + 16LL * v32);
          v33 = *v21;
          if ( a2 == *v21 )
            goto LABEL_36;
          ++v34;
        }
        if ( v35 )
          v21 = v35;
      }
      goto LABEL_36;
    }
    goto LABEL_68;
  }
LABEL_21:
  (*(void (__fastcall **)(__int64))(*(_QWORD *)result + 8LL))(result);
  result = v23[1];
LABEL_22:
  if ( v54[0].m128i_i64[0] )
  {
    v45 = result;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v54[0].m128i_i64[0] + 8LL))(v54[0].m128i_i64[0]);
    result = v45;
  }
  if ( (_BYTE *)v55[0] != v56 )
  {
    v46 = result;
    j_j___libc_free_0(v55[0]);
    return v46;
  }
  return result;
}

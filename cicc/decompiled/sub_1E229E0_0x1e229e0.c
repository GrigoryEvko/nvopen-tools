// Function: sub_1E229E0
// Address: 0x1e229e0
//
unsigned __int64 __fastcall sub_1E229E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v7; // esi
  __int64 v8; // rcx
  __int64 v9; // rdi
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r10
  unsigned __int64 result; // rax
  int v14; // r14d
  __int64 *v15; // r9
  int v16; // eax
  int v17; // edx
  __int64 v18; // rax
  unsigned __int64 *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rdi
  unsigned int v22; // edx
  __int64 v23; // r10
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rdi
  unsigned int v27; // ecx
  __int64 v28; // r10
  __int64 v30; // rax
  unsigned __int64 *v31; // rax
  unsigned int v32; // esi
  __int64 *v33; // r9
  int v34; // r14d
  int v35; // eax
  int v36; // r14d
  unsigned __int64 v37; // r9
  int v38; // eax
  __int64 v39; // [rsp+8h] [rbp-38h] BYREF
  __int64 v40; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v41[5]; // [rsp+18h] [rbp-28h] BYREF

  v39 = a2;
  v7 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
  {
    ++*(_QWORD *)a3;
LABEL_51:
    v7 *= 2;
    goto LABEL_52;
  }
  v8 = v39;
  v9 = *(_QWORD *)(a3 + 8);
  v10 = (v7 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
  v11 = (__int64 *)(v9 + 16LL * v10);
  v12 = *v11;
  if ( v39 == *v11 )
  {
LABEL_3:
    result = *((unsigned int *)v11 + 2);
    if ( (_DWORD)result )
      return result;
    goto LABEL_14;
  }
  v14 = 1;
  v15 = 0;
  while ( v12 != -8 )
  {
    if ( !v15 && v12 == -16 )
      v15 = v11;
    v10 = (v7 - 1) & (v14 + v10);
    v11 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( v39 == *v11 )
      goto LABEL_3;
    ++v14;
  }
  if ( !v15 )
    v15 = v11;
  v16 = *(_DWORD *)(a3 + 16);
  ++*(_QWORD *)a3;
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v7 )
    goto LABEL_51;
  if ( v7 - *(_DWORD *)(a3 + 20) - v17 <= v7 >> 3 )
  {
LABEL_52:
    sub_1DFB9D0(a3, v7);
    sub_1E1F250(a3, &v39, v41);
    v15 = (__int64 *)v41[0];
    v8 = v39;
    v17 = *(_DWORD *)(a3 + 16) + 1;
  }
  *(_DWORD *)(a3 + 16) = v17;
  if ( *v15 != -8 )
    --*(_DWORD *)(a3 + 20);
  *v15 = v8;
  *((_DWORD *)v15 + 2) = 0;
LABEL_14:
  v18 = (unsigned int)(*(_DWORD *)(a1 + 1040) - 1);
  *(_DWORD *)(a1 + 1040) = v18;
  v19 = (unsigned __int64 *)(*(_QWORD *)(a1 + 1032) + 48 * v18);
  if ( (unsigned __int64 *)*v19 != v19 + 2 )
    _libc_free(*v19);
  while ( 1 )
  {
    v32 = *(_DWORD *)(a4 + 24);
    if ( !v32 )
    {
      ++*(_QWORD *)a4;
      goto LABEL_26;
    }
    v20 = v39;
    v21 = *(_QWORD *)(a4 + 8);
    v22 = (v32 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
    result = v21 + 16LL * v22;
    v23 = *(_QWORD *)result;
    if ( *(_QWORD *)result != v39 )
    {
      v34 = 1;
      v33 = 0;
      while ( v23 != -8 )
      {
        if ( !v33 && v23 == -16 )
          v33 = (__int64 *)result;
        v22 = (v32 - 1) & (v34 + v22);
        result = v21 + 16LL * v22;
        v23 = *(_QWORD *)result;
        if ( v39 == *(_QWORD *)result )
          goto LABEL_17;
        ++v34;
      }
      if ( !v33 )
        v33 = (__int64 *)result;
      v35 = *(_DWORD *)(a4 + 16);
      ++*(_QWORD *)a4;
      result = (unsigned int)(v35 + 1);
      if ( 4 * (int)result < 3 * v32 )
      {
        if ( v32 - ((_DWORD)result + *(_DWORD *)(a4 + 20)) > v32 >> 3 )
          goto LABEL_28;
        goto LABEL_27;
      }
LABEL_26:
      v32 *= 2;
LABEL_27:
      sub_1E22820(a4, v32);
      sub_1E1F300(a4, &v39, v41);
      v33 = (__int64 *)v41[0];
      v20 = v39;
      result = (unsigned int)(*(_DWORD *)(a4 + 16) + 1);
LABEL_28:
      *(_DWORD *)(a4 + 16) = result;
      if ( *v33 != -8 )
        --*(_DWORD *)(a4 + 20);
      *v33 = v20;
      v33[1] = 0;
      return result;
    }
LABEL_17:
    v24 = *(_QWORD *)(result + 8);
    v40 = v24;
    if ( !v24 )
      return result;
    v25 = *(_DWORD *)(a3 + 24);
    if ( !v25 )
    {
      ++*(_QWORD *)a3;
      goto LABEL_48;
    }
    v26 = *(_QWORD *)(a3 + 8);
    v27 = (v25 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    result = v26 + 16LL * v27;
    v28 = *(_QWORD *)result;
    if ( v24 != *(_QWORD *)result )
      break;
LABEL_20:
    if ( (*(_DWORD *)(result + 8))-- != 1 )
      return result;
    v30 = (unsigned int)(*(_DWORD *)(a1 + 1040) - 1);
    *(_DWORD *)(a1 + 1040) = v30;
    v31 = (unsigned __int64 *)(*(_QWORD *)(a1 + 1032) + 48 * v30);
    if ( (unsigned __int64 *)*v31 != v31 + 2 )
      _libc_free(*v31);
    v39 = v40;
  }
  v36 = 1;
  v37 = 0;
  while ( v28 != -8 )
  {
    if ( v28 == -16 && !v37 )
      v37 = result;
    v27 = (v25 - 1) & (v36 + v27);
    result = v26 + 16LL * v27;
    v28 = *(_QWORD *)result;
    if ( v24 == *(_QWORD *)result )
      goto LABEL_20;
    ++v36;
  }
  if ( !v37 )
    v37 = result;
  v38 = *(_DWORD *)(a3 + 16);
  ++*(_QWORD *)a3;
  result = (unsigned int)(v38 + 1);
  if ( 4 * (int)result >= 3 * v25 )
  {
LABEL_48:
    v25 *= 2;
  }
  else if ( v25 - ((_DWORD)result + *(_DWORD *)(a3 + 20)) > v25 >> 3 )
  {
    goto LABEL_44;
  }
  sub_1DFB9D0(a3, v25);
  sub_1E1F250(a3, &v40, v41);
  v37 = v41[0];
  v24 = v40;
  result = (unsigned int)(*(_DWORD *)(a3 + 16) + 1);
LABEL_44:
  *(_DWORD *)(a3 + 16) = result;
  if ( *(_QWORD *)v37 != -8 )
    --*(_DWORD *)(a3 + 20);
  *(_QWORD *)v37 = v24;
  *(_DWORD *)(v37 + 8) = -1;
  return result;
}

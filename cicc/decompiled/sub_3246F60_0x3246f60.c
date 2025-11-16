// Function: sub_3246F60
// Address: 0x3246f60
//
unsigned __int64 __fastcall sub_3246F60(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  int v7; // eax
  unsigned int v8; // r8d
  unsigned __int64 *v9; // r9
  unsigned __int64 result; // rax
  size_t *v11; // rdi
  size_t v12; // rax
  unsigned __int64 v13; // rbx
  size_t v14; // rdx
  _BYTE *v15; // rcx
  unsigned int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdx
  _QWORD *v20; // rbx
  __int64 v21; // r12
  _BYTE *v22; // rax
  __int64 v23; // rax
  unsigned __int64 *v24; // [rsp+0h] [rbp-70h]
  unsigned __int64 *v25; // [rsp+0h] [rbp-70h]
  unsigned int v26; // [rsp+Ch] [rbp-64h]
  unsigned int v27; // [rsp+Ch] [rbp-64h]
  const void *v28; // [rsp+10h] [rbp-60h] BYREF
  size_t v29; // [rsp+18h] [rbp-58h]
  __m128i v30; // [rsp+20h] [rbp-50h] BYREF
  __int64 v31; // [rsp+30h] [rbp-40h]

  v28 = a3;
  v29 = a4;
  v30 = 0u;
  LODWORD(v31) = 0;
  v7 = sub_C92610();
  v8 = sub_C92740(a1, a3, a4, v7);
  v9 = (unsigned __int64 *)(*(_QWORD *)a1 + 8LL * v8);
  result = *v9;
  if ( *v9 )
  {
    if ( result != -8 )
      return result;
    --*(_DWORD *)(a1 + 16);
  }
  v11 = *(size_t **)(a1 + 24);
  v12 = *v11;
  v11[10] += a4 + 33;
  v13 = (v12 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v14 = a4 + 33 + v13;
  if ( v11[1] >= v14 && v12 )
  {
    *v11 = v14;
    v15 = (_BYTE *)(v13 + 32);
    if ( !a4 )
    {
      *v15 = 0;
      if ( !v13 )
        goto LABEL_9;
      goto LABEL_8;
    }
  }
  else
  {
    v25 = v9;
    v27 = v8;
    v23 = sub_9D1E70((__int64)v11, a4 + 33, a4 + 33, 3);
    v8 = v27;
    v9 = v25;
    v13 = v23;
    v15 = (_BYTE *)(v23 + 32);
    if ( !a4 )
    {
      *(_BYTE *)(v23 + 32) = 0;
      goto LABEL_8;
    }
  }
  v24 = v9;
  v26 = v8;
  v22 = memcpy(v15, a3, a4);
  v9 = v24;
  v8 = v26;
  v22[a4] = 0;
  if ( v13 )
  {
LABEL_8:
    *(_QWORD *)v13 = a4;
    *(__m128i *)(v13 + 8) = _mm_loadu_si128(&v30);
    *(_QWORD *)(v13 + 24) = v31;
  }
LABEL_9:
  *v9 = v13;
  ++*(_DWORD *)(a1 + 12);
  v16 = sub_C929D0((__int64 *)a1, v8);
  v19 = *(_QWORD *)a1;
  v20 = (_QWORD *)(*(_QWORD *)a1 + 8LL * v16);
  v21 = *v20;
  if ( !*v20 || v21 == -8 )
  {
    do
    {
      do
      {
        v21 = v20[1];
        ++v20;
      }
      while ( v21 == -8 );
    }
    while ( !v21 );
  }
  *(_DWORD *)(v21 + 24) = -1;
  *(_QWORD *)(v21 + 16) = *(_QWORD *)(a1 + 48);
  if ( *(_BYTE *)(a1 + 60) )
  {
    LOWORD(v31) = 261;
    v28 = *(const void **)(a1 + 32);
    v29 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)(v21 + 8) = sub_31DCC50(a2, (__int64 *)&v28, v19, v17, v18);
  }
  else
  {
    *(_QWORD *)(v21 + 8) = 0;
  }
  *(_QWORD *)(a1 + 48) += a4 + 1;
  return *v20;
}

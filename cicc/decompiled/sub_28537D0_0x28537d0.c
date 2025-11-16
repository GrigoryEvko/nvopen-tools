// Function: sub_28537D0
// Address: 0x28537d0
//
void __fastcall sub_28537D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // r15
  __int64 *v8; // r12
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rsi
  _QWORD *v13; // rax
  unsigned __int64 v14; // r10
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // r13
  char *v19; // r14
  __int64 v20; // r13
  int v21; // eax
  void *v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  size_t v25; // r10
  __int64 v26; // r14
  char *v27; // r14
  __int64 v28; // rax
  _QWORD *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // r14
  size_t v32; // [rsp+0h] [rbp-80h]
  __int64 v33; // [rsp+0h] [rbp-80h]
  const void *v34; // [rsp+8h] [rbp-78h]
  void *v35; // [rsp+8h] [rbp-78h]
  __int64 v36; // [rsp+10h] [rbp-70h]
  void *src; // [rsp+28h] [rbp-58h] BYREF
  _BYTE *v38; // [rsp+30h] [rbp-50h] BYREF
  __int64 v39; // [rsp+38h] [rbp-48h]
  _BYTE v40[64]; // [rsp+40h] [rbp-40h] BYREF

  v7 = *(__int64 **)(a1 + 64);
  v38 = v40;
  v39 = 0x200000000LL;
  v8 = &v7[*(unsigned int *)(a1 + 72)];
  if ( v7 != v8 )
  {
    v34 = (const void *)(a3 + 16);
    do
    {
      while ( 1 )
      {
        v11 = *(unsigned int *)(a3 + 8);
        v12 = *(_QWORD *)a3 + 8 * v11;
        v13 = sub_284FE40(*(_QWORD **)a3, v12, v7);
        v15 = a5 + 1;
        if ( (_QWORD *)v12 != v13 )
          break;
        if ( v15 > v14 )
        {
          sub_C8D5F0((__int64)&v38, v40, v15, 8u, a5, a6);
          a5 = (unsigned int)v39;
        }
        *(_QWORD *)&v38[8 * a5] = v11;
        v16 = *(unsigned int *)(a3 + 8);
        v17 = *(unsigned int *)(a3 + 12);
        LODWORD(v39) = v39 + 1;
        v18 = *v7;
        if ( v16 + 1 > v17 )
        {
          sub_C8D5F0(a3, v34, v16 + 1, 8u, a5, a6);
          v16 = *(unsigned int *)(a3 + 8);
        }
        ++v7;
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v16) = v18;
        ++*(_DWORD *)(a3 + 8);
        if ( v8 == v7 )
          goto LABEL_12;
      }
      v10 = ((__int64)v13 - a6) >> 3;
      if ( v15 > v14 )
      {
        v33 = v10;
        sub_C8D5F0((__int64)&v38, v40, v15, 8u, a5, a6);
        a5 = (unsigned int)v39;
        v10 = v33;
      }
      ++v7;
      *(_QWORD *)&v38[8 * a5] = v10;
      LODWORD(v39) = v39 + 1;
    }
    while ( v8 != v7 );
  }
LABEL_12:
  v19 = *(char **)a1;
  v20 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  for ( src = v19; (char *)v20 != v19; src = v19 )
  {
    if ( *(_QWORD *)v19 == 4101 )
    {
      v28 = *(unsigned int *)(a2 + 8);
      if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), v28 + 1, 8u, a5, a6);
        v28 = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v28) = 4101;
      v29 = src;
      v30 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v30;
      v31 = *(_QWORD *)&v38[8 * v29[1]];
      if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, (const void *)(a2 + 16), v30 + 1, 8u, a5, a6);
        v30 = *(unsigned int *)(a2 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v30) = v31;
      ++*(_DWORD *)(a2 + 8);
    }
    else
    {
      v21 = sub_AF4160((unsigned __int64 **)&src);
      v22 = src;
      v23 = (__int64)&v19[8 * v21];
      v24 = *(unsigned int *)(a2 + 8);
      v25 = v23 - (_QWORD)src;
      v26 = (v23 - (__int64)src) >> 3;
      if ( v26 + v24 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v32 = v23 - (_QWORD)src;
        v35 = src;
        v36 = v23;
        sub_C8D5F0(a2, (const void *)(a2 + 16), v26 + v24, 8u, (__int64)src, v23);
        v24 = *(unsigned int *)(a2 + 8);
        v25 = v32;
        v22 = v35;
        v23 = v36;
      }
      if ( (void *)v23 != v22 )
      {
        memcpy((void *)(*(_QWORD *)a2 + 8 * v24), v22, v25);
        LODWORD(v24) = *(_DWORD *)(a2 + 8);
      }
      *(_DWORD *)(a2 + 8) = v24 + v26;
    }
    v27 = (char *)src;
    v19 = &v27[8 * (unsigned int)sub_AF4160((unsigned __int64 **)&src)];
  }
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
}
